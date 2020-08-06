"""Tests for third_party.deepmath.deephol.io_util."""
import tensorflow.compat.v1 as tf
from deepmath.deephol import deephol_pb2
from deepmath.deephol import io_util


class IOUtilTest(tf.test.TestCase):

  def setUp(self):
    super(IOUtilTest, self).setUp()
    self.tac0 = deephol_pb2.Tactic(id=0, name="TAC0")
    self.tac1 = deephol_pb2.Tactic(id=1, name="TAC1")
    self.tac1r = deephol_pb2.Tactic(id=1, name="TAC1r")
    self.tac2 = deephol_pb2.Tactic(id=2, name="TAC2")
    self.tactics_info = deephol_pb2.TacticsInfo()
    self.replacements = deephol_pb2.TacticsInfo()

  def test_process_tactics_with_correct_input(self):
    self.tactics_info.tactics.extend([self.tac0, self.tac1])
    actual = io_util._process_tactics_and_replacements(self.tactics_info,
                                                       self.replacements)
    self.assertEqual([self.tac0, self.tac1], actual)

  def test_process_tactics_with_gap(self):
    self.tactics_info.tactics.extend([self.tac0, self.tac2])
    with self.assertRaisesRegexp(ValueError, "should not have id"):
      io_util._process_tactics_and_replacements(self.tactics_info,
                                                self.replacements)

  def test_process_tactics_unordered_fails(self):
    self.tactics_info.tactics.extend([self.tac1, self.tac0])
    with self.assertRaisesRegexp(ValueError, "should not have id"):
      io_util._process_tactics_and_replacements(self.tactics_info,
                                                self.replacements)

  def test_process_tactics_replacement_with_correct_input(self):
    self.tactics_info.tactics.extend([self.tac0, self.tac1])
    self.replacements.tactics.extend([self.tac1r])
    actual = io_util._process_tactics_and_replacements(self.tactics_info,
                                                       self.replacements)
    self.assertEqual([self.tac0, self.tac1r], actual)

  def test_process_tactics_out_of_range_replacement_fails(self):
    self.tactics_info.tactics.extend([self.tac0])
    self.replacements.tactics.extend([self.tac1r])
    with self.assertRaisesRegexp(ValueError, "Replacement.*bounds"):
      io_util._process_tactics_and_replacements(self.tactics_info,
                                                self.replacements)

  def test_read_proof_logs(self):
    tempfile = self.create_tempfile(
        file_path="proof_log.textpb", content="num_proofs: 0")
    proof_log_iterator = io_util.read_proof_logs(tempfile.full_path)
    proof_logs = [p for p in proof_log_iterator]
    self.assertEqual([deephol_pb2.ProofLog(num_proofs=0)], proof_logs)


if __name__ == "__main__":
  tf.test.main()
