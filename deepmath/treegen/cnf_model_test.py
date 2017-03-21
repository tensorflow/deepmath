# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for treegen.cnf_model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import tensorflow as tf
from deepmath.treegen import cnf_model
from deepmath.treegen import cnf_model_test_lib

flags = tf.flags
FLAGS = flags.FLAGS


class CnfModelTest(tf.test.TestCase):
  # From l102_finseq_1, in test0.jsonl
  # v7_ordinal1(X1) | ~m1_subset_1(X1, k4_ordinal1())
  tiny_expr = json.loads(
      '''{"clauses": [{"positive": true, "params": [{"var": "X1"}], "pred":
      "v7_ordinal1"}, {"positive": false, "params": [{"var": "X1"}, {"params":
      [], "func": "k4_ordinal1"}], "pred": "m1_subset_1"}]}''')

  # From l102_modelc_2, in test0.jsonl
  huge_expr = json.loads(
      '''{"clauses": [{"positive": true, "params": [{"params": [], "func":
      "esk4_0"}, {"var": "X1"}, {"var": "X1"}, {"var": "X1"}], "pred":
      "r5_modelc_2"}, {"positive": false, "equal": [{"params": [{"var": "X1"},
      {"params": [{"params": [], "func": "esk4_0"}, {"var": "X1"}, {"var":
      "X2"}, {"var": "X3"}], "func": "esk2_4"}], "func": "k1_funct_1"},
      {"params": [{"var": "X1"}, {"params": [{"params": [], "func": "esk4_0"},
      {"var": "X1"}, {"var": "X2"}, {"var": "X3"}], "func": "esk2_4"}], "func":
      "k1_funct_1"}]}, {"positive": false, "params": [{"var": "X1"}, {"params":
      [{"params": [], "func": "esk4_0"}, {"var": "X1"}, {"var": "X2"}, {"var":
      "X3"}], "func": "esk2_4"}, {"params": [], "func": "esk4_0"}], "pred":
      "epred2_3"}, {"positive": false, "params": [{"var": "X1"}], "pred":
      "v7_ordinal1"}, {"positive": false, "params": [{"params": [{"params": [],
      "func": "esk4_0"}, {"var": "X1"}, {"var": "X2"}, {"var": "X3"}], "func":
      "esk2_4"}], "pred": "v3_modelc_2"}, {"positive": false, "params":
      [{"params": [{"params": [], "func": "esk4_0"}, {"var": "X1"}, {"var":
      "X2"}, {"var": "X3"}], "func": "esk2_4"}, {"params": [], "func":
      "k5_numbers"}], "pred": "m2_finseq_1"}, {"positive": false, "params":
      [{"params": [{"params": [], "func": "esk4_0"}, {"var": "X1"}, {"var":
      "X2"}, {"var": "X3"}], "func": "esk2_4"}], "pred": "v1_modelc_2"},
      {"positive": false, "params": [{"params": [], "func": "esk4_0"},
      {"params": [], "func": "esk5_0"}, {"var": "X1"}], "pred": "r4_modelc_2"},
      {"positive": false, "params": [{"var": "X1"}, {"params": [{"params":
      [{"params": [], "func": "k9_modelc_2"}, {"params": [{"params": [], "func":
      "esk4_0"}], "func": "u1_struct_0"}], "func": "k2_zfmisc_1"}], "func":
      "k1_zfmisc_1"}], "pred": "m1_subset_1"}, {"positive": false, "params":
      [{"var": "X1"}, {"params": [{"params": [{"params": [], "func":
      "k15_modelc_2"}, {"params": [{"params": [], "func": "esk4_0"}], "func":
      "u1_modelc_2"}], "func": "k2_zfmisc_1"}], "func": "k1_zfmisc_1"}], "pred":
      "m1_subset_1"}, {"positive": false, "params": [{"var": "X1"}, {"params":
      [], "func": "k9_modelc_2"}, {"params": [{"params": [], "func": "esk4_0"}],
      "func": "u1_struct_0"}], "pred": "v1_funct_2"}, {"positive": false,
      "params": [{"var": "X1"}, {"params": [], "func": "k15_modelc_2"},
      {"params": [{"params": [], "func": "esk4_0"}], "func": "u1_modelc_2"}],
      "pred": "v1_funct_2"}, {"positive": false, "params": [{"var": "X1"}],
      "pred": "v1_funct_1"}, {"positive": false, "params": [{"var": "X1"}],
      "pred": "v1_funct_1"}]}''')

  def testSeqModelMemorizesTinyExpr(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        num_iterations=200,
        extra_hparams='depth=1',
        model_class=cnf_model.CNFSequenceModel)

  def testSeqModelMemorizesTinyExprMaskedXent(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        num_iterations=200,
        extra_hparams='depth=1,masked_xent=true',
        model_class=cnf_model.CNFSequenceModel)

  def testSeqModelWorksWithTinyHugeExpr(self):
    cnf_model_test_lib.test_memorization(
        self, [self.tiny_expr, self.huge_expr],
        num_iterations=1,
        model_class=cnf_model.CNFSequenceModel)

  def testSeqModelWorksWithTinyHugeExprMaskedXent(self):
    cnf_model_test_lib.test_memorization(
        self, [self.tiny_expr, self.huge_expr],
        num_iterations=1,
        extra_hparams='masked_xent=true',
        model_class=cnf_model.CNFSequenceModel)

  def testTreeModelMemorizesTinyExprStdFixedZ(self):
    cnf_model_test_lib.test_memorization(self, self.tiny_expr)

  def testTreeModelMemorizesTinyExprStdVae(self):
    cnf_model_test_lib.test_memorization(
        self, self.tiny_expr, extra_hparams='objective=vae,min_kl_weight=1')

  def testTreeModelMemorizesTinyExprStdIwaeMcSamples2(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        extra_hparams='objective=iwae,min_kl_weight=1,mc_samples=2')

  def testTreeModelMemorizesTinyExprStdVaeMix(self):
    cnf_model_test_lib.test_memorization(
        self, self.tiny_expr, extra_hparams='objective=vae_mix,batch_size=3',
        num_iterations=150)

  def testTreeModelMemorizesTinyExprAuxLstmFixedZ(self):
    cnf_model_test_lib.test_memorization(
        self, self.tiny_expr, extra_hparams='model_variants=[aux_lstm]')

  def testTreeModelMemorizesTinyExprUncondSibFixedZ(self):
    cnf_model_test_lib.test_memorization(
        self, self.tiny_expr, extra_hparams='model_variants=[uncond_sib]')

  def testTreeModelMemorizesTinyExprGatedSigmoidFixedZ(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        extra_hparams='model_variants=[gated],gate_type=sigmoid')

  def testTreeModelMemorizesTinyExprGatedSoftmaxFixedZ(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        extra_hparams='model_variants=[gated],gate_type=softmax')

  def testTreeModelMemorizesTinyExprGatedTiedFixedZ(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        extra_hparams='model_variants=[gated],gate_tied=true')

  def testTreeModelMemorizesTinyExprAuxLstmUncondSibFixedZ(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        extra_hparams='model_variants=[aux_lstm,uncond_sib]')

  def testTreeModelMemorizesTinyExprAuxLstmVae(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        num_iterations=80,
        extra_hparams='model_variants=[aux_lstm],objective=vae,min_kl_weight=1')

  def testTreeModelMemorizesTinyExprAuxLstmGatedUncondSibFixedZ(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        extra_hparams='model_variants=[aux_lstm,gated,uncond_sib]')

  def testTreeModelMemorizesTinyExprGatedUncondSibVae(self):
    cnf_model_test_lib.test_memorization(
        self, self.tiny_expr, extra_hparams='model_variants=[gated,uncond_sib]')

  def testTreeModelMemorizesTinyExprAuxLstmGatedLayerNormUncondSibVae(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        extra_hparams='model_variants=[aux_lstm,gated,layer_norm,uncond_sib]')

  def testTreeModelMemorizesTinyExprGatedLayerNormUncondSibVae(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        extra_hparams='model_variants=[gated,layer_norm,uncond_sib]')

  def testTreeModelWorksWithTinyExprTanhMostVariationssVae(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        num_iterations=1,
        extra_hparams='model_variants=[gated,layer_norm,rev_read,uncond_sib],'
        'act_fn=tanh,objective=vae,min_kl_weight=1')

  def testTreeModelMemorizesTinyExprMostVariationsDeepFixedZ(self):
    cnf_model_test_lib.test_memorization(
        self,
        self.tiny_expr,
        extra_hparams='model_variants=[gated,layer_norm,rev_read,uncond_sib],'
        'highway_layers=5,op_hidden=1')


if __name__ == '__main__':
  tf.test.main()
