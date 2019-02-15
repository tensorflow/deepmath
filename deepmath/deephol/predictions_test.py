"""Tests for predictions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from deepmath.deephol import mock_predictions_lib
from deepmath.deephol import predictions

TEST_ARRAY = np.reshape(np.arange(100), (10, 10)).astype(float)
MOCK_PREDICTOR = mock_predictions_lib.MockPredictionsLib


def double(x):
  if x is None:
    return x
  else:
    return 2 * x


class PredictionsTest(tf.test.TestCase, parameterized.TestCase):

  def test_batch_array_with_none(self):
    result = predictions.batch_array(TEST_ARRAY, None)
    self.assertEqual(len(result), 1)
    self.assertAllEqual(TEST_ARRAY, result[0])

  def test_batch_array_with_batch_size_1(self):
    result = predictions.batch_array(TEST_ARRAY, 1)
    self.assertEqual(len(result), 10)
    for i in range(10):
      self.assertAllEqual(np.expand_dims(TEST_ARRAY[i, :], 0), result[i])

  def test_batch_array_with_batch_size_3(self):
    result = predictions.batch_array(TEST_ARRAY, 3)
    expected = [
        TEST_ARRAY[:3, :], TEST_ARRAY[3:6, :], TEST_ARRAY[6:9, :],
        TEST_ARRAY[9:, :]
    ]
    self.assertEqual(len(result), len(expected))
    for i in range(len(expected)):
      self.assertAllEqual(expected[i], result[i])

  def test_batch_array_with_batch_size_10(self):
    result = predictions.batch_array(TEST_ARRAY, 10)
    self.assertEqual(len(result), 1)
    self.assertAllEqual(TEST_ARRAY, result[0])

  def test_batch_array_with_batch_size_15(self):
    result = predictions.batch_array(TEST_ARRAY, 15)
    self.assertEqual(len(result), 1)
    self.assertAllEqual(TEST_ARRAY, result[0])

  def test_batch_array_strlist_with_batch_size_3(self):
    strlist = [str(i) for i in range(10)]
    result = predictions.batch_array(strlist, 3)
    expected = [strlist[:3], strlist[3:6], strlist[6:9], [strlist[9]]]
    print('result:', result)
    self.assertEqual(len(expected), len(result))
    for i in range(len(expected)):
      self.assertAllEqual(expected[i], result[i])

  def test_batch_array_strlist_with_batch_size_none(self):
    strlist = [str(i) for i in range(10)]
    result = predictions.batch_array(strlist, None)
    self.assertEqual(len(result), 1)
    self.assertAllEqual(result[0], strlist)

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_batched_run_identity(self, max_batch_size):
    result = predictions.batched_run([TEST_ARRAY], lambda x: x, max_batch_size)
    self.assertAllEqual(result, TEST_ARRAY)

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_batched_run_add(self, max_batch_size):
    result = predictions.batched_run(
        [TEST_ARRAY, TEST_ARRAY], lambda x, y: x + y, max_batch_size)
    self.assertAllEqual(result, 2.0 * TEST_ARRAY)

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_batched_run_str_to_int_and_back(self, max_batch_size):
    strlist = [str(i) for i in range(10)]
    result = predictions.batched_run(
        [strlist], lambda l: np.array([[float(x)] for x in l]), max_batch_size)
    self.assertAllEqual(result, [[float(i)] for i in range(10)])

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_predict_goal_embedding(self, max_batch_size):
    predictor = MOCK_PREDICTOR(max_batch_size, double(max_batch_size))
    self.assertAllEqual(
        predictor.goal_embedding('goal'),
        predictor.batch_goal_embedding(['goal'])[0])

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_predict_thm_embedding(self, max_batch_size):
    predictor = MOCK_PREDICTOR(max_batch_size, double(max_batch_size))
    self.assertAllEqual(
        predictor.thm_embedding('thm'),
        predictor.batch_thm_embedding(['thm'])[0])

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_predict_batch_goal_embedding(self, max_batch_size):
    strlist = [str(i) for i in range(10)]
    predictor = MOCK_PREDICTOR(max_batch_size, double(max_batch_size))
    self.assertAllEqual(
        predictor.batch_goal_embedding(strlist),
        predictor._batch_goal_embedding(strlist))

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_predict_batch_thm_embedding(self, max_batch_size):
    strlist = [str(i) for i in range(10)]
    predictor = MOCK_PREDICTOR(max_batch_size, double(max_batch_size))
    self.assertAllEqual(
        predictor.batch_thm_embedding(strlist),
        predictor._batch_thm_embedding(strlist))

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_predict_batch_tactic_scores(self, max_batch_size):
    predictor = MOCK_PREDICTOR(max_batch_size, double(max_batch_size))
    self.assertAllEqual(
        predictor.batch_tactic_scores(TEST_ARRAY),
        predictor._batch_tactic_scores(TEST_ARRAY))

  @parameterized.parameters(1, 2, 3, 10, 15, None)
  def test_predict_batch_thm_scores(self, max_batch_size):
    predictor = MOCK_PREDICTOR(max_batch_size, double(max_batch_size))
    state = np.arange(10)
    dup_state = np.tile(np.arange(10), [10, 1])
    self.assertAllEqual(
        predictor.batch_thm_scores(state, TEST_ARRAY),
        predictor._batch_thm_scores(dup_state, TEST_ARRAY))
    self.assertAllEqual(
        predictor.batch_thm_scores(state, TEST_ARRAY, tactic_id=4),
        predictor._batch_thm_scores(dup_state, TEST_ARRAY, tactic_id=4))


if __name__ == '__main__':
  tf.test.main()
