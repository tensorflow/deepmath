"""Converter from ProofLog proto to a tf Example."""

from __future__ import absolute_import
from __future__ import division
# Import Type Annotations
from __future__ import print_function
import tensorflow as tf
from tf import logging
from typing import Dict, Iterable, List, Optional, Text, Tuple
from deepmath.deephol import deephol_pb2
from deepmath.deephol import io_util
from deepmath.deephol import theorem_fingerprint as fp
from deepmath.deephol.deephol_loop import options_pb2
from deepmath.proof_assistant import proof_assistant_pb2

BYTES_FEATURE_ENCODING = 'utf-8'


class ProofLogToTFExample(object):
  """TFExampleFormat.HOLPARAM conversion to TFExamples."""

  def __init__(self, tactic_name_id_map: Dict[Text, int],
               theorem_database: proof_assistant_pb2.TheoremDatabase,
               options: options_pb2.ConvertorOptions):
    """Initializer.

    Arguments:
      tactic_name_id_map: mapping from tactic names to ids.
      theorem_database: database containing list of global theorems with splits
      options: options to control forbidden parameters and graph representations
    """
    if options.scrub_parameters not in [
        options_pb2.ConvertorOptions.NOTHING,
        options_pb2.ConvertorOptions.TESTING,
        options_pb2.ConvertorOptions.VALIDATION_AND_TESTING
    ]:
      raise ValueError('Unknown scrub_parameter.')

    self.tactic_name_id_map = tactic_name_id_map
    self.options = options

    self.fingerprint_conclusion_map = {
        fp.Fingerprint(theorem): theorem.conclusion
        for theorem in theorem_database.theorems
    }

    self.forbidden_parameters = set()
    for theorem in theorem_database.theorems:
      if (theorem.tag not in [
          proof_assistant_pb2.Theorem.DEFINITION,
          proof_assistant_pb2.Theorem.TYPE_DEFINITION
      ] and theorem.training_split == proof_assistant_pb2.Theorem.UNKNOWN):
        raise ValueError('need training split information in theorem database.')

      scrub_testsplit_parameters = options.scrub_parameters in [
          options_pb2.ConvertorOptions.TESTING,
          options_pb2.ConvertorOptions.VALIDATION_AND_TESTING
      ]
      scrub_validsplit_parameters = (
          options.scrub_parameters ==
          options_pb2.ConvertorOptions.VALIDATION_AND_TESTING)

      if ((theorem.training_split == proof_assistant_pb2.Theorem.TESTING and
           scrub_testsplit_parameters) or
          (theorem.training_split == proof_assistant_pb2.Theorem.VALIDATION and
           scrub_validsplit_parameters)):
        self.forbidden_parameters.add(fp.Fingerprint(theorem))

  def _int64_feature(self, ints):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=ints))

  def _bytes_feature(self, strings: Iterable[Text]):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[s.encode(encoding=BYTES_FEATURE_ENCODING) for s in strings]))

  def _get_parameter_conclusion(self,
                                parameter: proof_assistant_pb2.Theorem) -> Text:
    """Get conclusion from fingerprint (prioritized), or one in theorem."""
    if (parameter.HasField('fingerprint') and
        parameter.fingerprint in self.fingerprint_conclusion_map):
      conclusion = self.fingerprint_conclusion_map[parameter.fingerprint]
      if (parameter.HasField('conclusion') and
          conclusion != parameter.conclusion):
        raise ValueError('conclusion doesn\'t match that in database')
      return conclusion
    if parameter.HasField('conclusion'):
      return parameter.conclusion
    raise ValueError('Neither conclusion present, nor fingerprint %d found'
                     ' in theorem database.' % parameter.fingerprint)

  def _extract_theorem_parameters(
      self, tactic_application: deephol_pb2.TacticApplication
  ) -> Tuple[List[Text], List[Text]]:
    """Extracts parameters of type theorem from a tactic application, if any.

    Note: it might be misleading to call these theorems. If the source is from
    an assumption, the theorem is of the form x |- x. We return x in this case.

    Arguments:
      tactic_application: tactic application to extract the parameters from.

    Returns:
      A pair of (parameters, hard_negatives), where parameters are the
      conclusions of the parameters and hard_negatives are those selected
      parameters that did not contribute to the final outcome. Both are
      preprocessed.
    """
    theorems = []
    hard_negatives = []
    for parameter in tactic_application.parameters:
      if parameter.theorems and not (
          parameter.parameter_type == deephol_pb2.Tactic.THEOREM or
          parameter.parameter_type == deephol_pb2.Tactic.THEOREM_LIST):
        raise ValueError('Unexpected theorem parameters or incorrect type.')

      theorems += [
          self._get_parameter_conclusion(theorem)
          for theorem in parameter.theorems
          if fp.Fingerprint(theorem) not in self.forbidden_parameters
      ]
      hard_negatives += [
          self._get_parameter_conclusion(theorem)
          for theorem in parameter.hard_negative_theorems
          if fp.Fingerprint(theorem) not in self.forbidden_parameters
      ]
    return theorems, hard_negatives

  def _get_thm_features(self, theorem_parameters: List[Text]
                       ) -> Dict[Text, tf.train.Feature]:
    """Constructs the features for the theorem parameters.

    Args:
      theorem_parameters: A list of sexpressions that represents the theorem
        arguments.

    Returns:
      A dictionary that maps the names of the features to tf.train.Feature
      objects.
    """
    thm_features = {
        # theorem parameters in tactic application
        'thms': self._bytes_feature(theorem_parameters),
    }
    return thm_features

  def _proof_step_features(self, goal_proto: proof_assistant_pb2.Theorem,
                           tactic_application: deephol_pb2.TacticApplication
                          ) -> Dict[Text, tf.train.Feature]:
    """Compute the basic features of a proof step (goal, tactic, and args)."""
    # preprocessed goal's conclusion's features
    features = {'goal': self._bytes_feature([goal_proto.conclusion])}
    tactic_id = self.tactic_name_id_map[tactic_application.tactic]
    theorem_parameters, hard_negatives = self._extract_theorem_parameters(
        tactic_application)
    features.update(self._get_thm_features(theorem_parameters))
    features.update({
        # preprocessed goal's hypotheses
        'goal_asl': self._bytes_feature(goal_proto.hypotheses),
        # tactic id of tactic application
        'tac_id': self._int64_feature([tactic_id]),
        # Hard (high scoring) negative examples for the parameters that were
        # selected specifically to train against.
        'thms_hard_negatives': self._bytes_feature(hard_negatives),
    })
    return features

  def process_proof_step(self, goal_proto: proof_assistant_pb2.Theorem,
                         tactic_application: deephol_pb2.TacticApplication
                        ) -> Optional[tf.train.Example]:
    """Convert goal,tactic pair to TFExample (for closed goal) or None."""
    if not tactic_application.closed:
      return None
    features = self._proof_step_features(goal_proto, tactic_application)
    return tf.train.Example(features=tf.train.Features(feature=features))

  def process_proof_node(self, proof_node: deephol_pb2.ProofNode):
    for tactic_application in proof_node.proofs:
      tfexample = self.process_proof_step(proof_node.goal, tactic_application)
      if tfexample is not None:
        yield tfexample

  def process_proof_log(self, proof_log: deephol_pb2.ProofLog):
    for proof_node in proof_log.nodes:
      for example in self.process_proof_node(proof_node):
        yield example

  def process_proof_logs(self, proof_logs):
    for proof_log in proof_logs:
      for example in self.process_proof_log(proof_log):
        yield example

  def to_negative_example(self, negative_theorem: proof_assistant_pb2.Theorem
                         ) -> tf.train.Example:
    raise NotImplementedError(
        'to_negative_example not implemented for base class.')


def create_processor(
    options: options_pb2.ConvertorOptions,
    theorem_database: Optional[proof_assistant_pb2.TheoremDatabase] = None,
    tactics: Optional[List[deephol_pb2.Tactic]] = None) -> ProofLogToTFExample:
  """Factory function for ProofLogToTFExample."""

  if theorem_database and options.theorem_database_path:
    raise ValueError(
        'Both thereom database as well as a path to load it from file '
        'provided. Only provide one.')
  if not theorem_database:
    theorem_database = io_util.load_theorem_database_from_file(
        str(options.theorem_database_path))

  if tactics and options.tactics_path:
    raise ValueError('Both tactics as well as a path to load it from '
                     'provided. Only provide one.')
  if not tactics:
    tactics = io_util.load_tactics_from_file(str(options.tactics_path), None)
  tactics_name_id_map = {tactic.name: tactic.id for tactic in tactics}

  if options.replacements_hack:
    logging.warning('Replacments hack is enabled.')
    tactics_name_id_map.update({
        'GEN_TAC': 8,
        'MESON_TAC': 11,
        'CHOOSE_TAC': 34,
    })
  if options.format != options_pb2.ConvertorOptions.HOLPARAM:
    raise ValueError('Unknown options_pb2.ConvertorOptions.TFExampleFormat.')
  return ProofLogToTFExample(tactics_name_id_map, theorem_database, options)
