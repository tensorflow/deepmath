# python3
"""Decoder-Encoder Architecture for Generic Sequence Translation."""

import copy
import json

from language.bert import modeling as bert_modeling
from language.bert import optimization
import numpy as np
import six
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
import tensorflow_probability as tfp


class TransformerConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               embedding_size=128,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               initializer_range=0.02,
               convolution_size=0,
               convolution_filters=128,
               deconvolution_size=0,
               deconvolution_filters=256,
               autoregressive_feedback_filters=256,
               autoregressive_feedback_weight=20.0,
               autoregressive_feedback_layers=0,
               autoregressive_feedback_type="masked"):
    """Constructs TransformerConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      embedding_size: Size of the word embedding vectors.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048). If 0, do not use position embeddings.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
      convolution_size: If nonzero, then two layers of convolution is used
        as the first two layers of the network: a stride one with patch_size=
          convolution_size and one with both stride and patch_size being
          convolution_size. The second convolution has both patch_size and
          stride = convolution_size.
      convolution_filters: The number of filters in the first convolutional
        layers if convolution_size > 0. Otherwise it is ignored.
      deconvolution_size: A deconvolution with this stride and patch_size is
        applied as the top layer of the transformer if deconvolution_size > 0.
      deconvolution_filters: The number of filters in the dine deconvolution
        layer if deconvolution_size > 0. Otherwise it is ignored.
      autoregressive_feedback_filters: If this is nonzero and deconvolution size
        is at least 2 then a 1x1 convolution is added to enable autoregressive
        processing.
      autoregressive_feedback_weight: If autoregressive feedback is enabled, its
        activations will be weighted by this value.
      autoregressive_feedback_layers: The number of extra layers added in order
        to combine the autoregressive feedback activations with the deconvolved
        transformer outputs.
      autoregressive_feedback_type: If autoregressive feedback is enabled, this
        can be either "masked", "conv" to choose the feedback mechanism. "none"
        disables the feedback meachanism.
    """
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.initializer_range = initializer_range
    self.convolution_size = convolution_size
    self.convolution_filters = convolution_filters
    self.deconvolution_size = deconvolution_size
    self.deconvolution_filters = deconvolution_filters
    self.autoregressive_feedback_filters = autoregressive_feedback_filters
    self.autoregressive_feedback_weight = autoregressive_feedback_weight
    self.autoregressive_feedback_layers = autoregressive_feedback_layers
    assert autoregressive_feedback_type in ["masked", "conv", "none"]
    self.autoregressive_feedback_type = autoregressive_feedback_type

  @property
  def use_position_embeddings(self) -> bool:
    return bool(self.max_position_embeddings)

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `TransformerConfig` from a Python dictionary of parameters."""
    config = TransformerConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `TransformerConfig` from a json file of parameters."""
    with tf.io.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def transformer_model(input_tensor,
                      attention_mask=None,
                      side_tensor=None,
                      side_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=bert_modeling.gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    side_tensor: (optional) float Tensor of shape [batch_size, side_seq_length,
      hidden_size].
    side_mask: (optional) int32 Tensor of shape [batch_size, side_seq_length]
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  tf.logging.info("Creating transformer...")
  orig_attention_mask_shape = bert_modeling.get_shape_list(
      attention_mask, expected_rank=[2, 3])
  tf.logging.info("orig attention_mask shape: %s", orig_attention_mask_shape)
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = bert_modeling.get_shape_list(input_tensor, expected_rank=3)
  tf.logging.info("orig input tensor shape: %s", input_shape)
  input_width = input_shape[2]
  if input_width != hidden_size:
    with tf.variable_scope("input_proj"):
      input_tensor = bert_modeling.dense_layer_2d(
          input_tensor, hidden_size,
          bert_modeling.create_initializer(initializer_range), None, "dense")
  input_shape = bert_modeling.get_shape_list(input_tensor, expected_rank=3)
  input_width = input_shape[2]
  tf.logging.info("input tensor shape: %s", input_shape)

  if side_tensor is not None:
    assert side_mask is not None
    side_shape = bert_modeling.get_shape_list(side_tensor, expected_rank=3)
    tf.logging.info("orig side tensor shape: %s", side_shape)
    side_mask_shape = bert_modeling.get_shape_list(side_mask, expected_rank=2)
    side_width = side_shape[2]
    assert side_mask_shape[0] == side_shape[0]
    assert side_mask_shape[1] == side_shape[1]
    assert input_shape[0] == side_shape[0]
    side_mask = tf.cast(tf.expand_dims(side_mask, 1), dtype=tf.float32)
    new_zeros = tf.zeros((1, input_shape[1], 1), dtype=tf.float32)
    side_mask = side_mask + new_zeros
    side_mask_shape = bert_modeling.get_shape_list(side_mask, expected_rank=3)
    attention_mask_shape = bert_modeling.get_shape_list(
        attention_mask, expected_rank=3)
    tf.logging.info("orig_attention_mask shape: %s", attention_mask_shape)
    tf.logging.info("side_mask shape: %s", side_mask_shape)
    attention_mask = tf.concat([side_mask, attention_mask], axis=2)
    attention_mask_shape = bert_modeling.get_shape_list(
        attention_mask, expected_rank=3)
    tf.logging.info("attention_mask shape: %s", attention_mask_shape)
    if side_width != hidden_size:
      with tf.variable_scope("side_proj"):
        side_tensor = bert_modeling.dense_layer_2d(
            side_tensor, hidden_size,
            bert_modeling.create_initializer(initializer_range), None, "dense")
    side_shape = bert_modeling.get_shape_list(side_tensor, expected_rank=3)
    tf.logging.info("input tensor shape: %s", input_shape)
    side_width = side_shape[2]
    if side_width != hidden_size:
      raise ValueError("The width of the side tensor (%d) != hidden size (%d)" %
                       (side_width, hidden_size))

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  prev_output = input_tensor
  all_layer_outputs = []
  to_tensors = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output
      layer_input_shape = bert_modeling.get_shape_list(
          layer_input, expected_rank=3)
      tf.logging.info("layer input shape: %s", layer_input_shape)
      with tf.variable_scope("attention"):
        with tf.variable_scope("self"):
          from_tensor = layer_input
          if side_tensor is not None:
            tf.logging.info("concatenating side tensor")
            to_tensor = tf.concat([side_tensor, layer_input], axis=1)
          else:
            tf.logging.info("no side tensor to concatenate")
            to_tensor = layer_input
          to_tensor_shape = bert_modeling.get_shape_list(
              to_tensor, expected_rank=3)
          to_tensors.append(to_tensor)
          from_tensor_shape = bert_modeling.get_shape_list(
              from_tensor, expected_rank=3)
          attention_mask_shape = bert_modeling.get_shape_list(
              attention_mask, expected_rank=[2, 3])
          tf.logging.info("from tensor shape: %s", from_tensor_shape)
          tf.logging.info("to tensor shape: %s", to_tensor_shape)
          tf.logging.info("attention mask shape: %s", attention_mask_shape)
          attention_output = bert_modeling.attention_layer(
              from_tensor=from_tensor,
              to_tensor=to_tensor,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = bert_modeling.dense_layer_3d_proj(
              attention_output, hidden_size, num_attention_heads,
              attention_head_size,
              bert_modeling.create_initializer(initializer_range), None,
              "dense")
          attention_output = bert_modeling.dropout(attention_output,
                                                   hidden_dropout_prob)
          attention_output = bert_modeling.layer_norm(attention_output +
                                                      layer_input)

          attention_output_shape = bert_modeling.get_shape_list(
              attention_output, expected_rank=3)
          tf.logging.info("attention output shape: %s", attention_output_shape)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = bert_modeling.dense_layer_2d(
            attention_output, intermediate_size,
            bert_modeling.create_initializer(initializer_range),
            intermediate_act_fn, "dense")
        intermediate_output_shape = bert_modeling.get_shape_list(
            intermediate_output, expected_rank=3)
        tf.logging.info("intermediate output shape: %s",
                        intermediate_output_shape)

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = bert_modeling.dense_layer_2d(
            intermediate_output, hidden_size,
            bert_modeling.create_initializer(initializer_range), None, "dense")
        layer_output = bert_modeling.dropout(layer_output, hidden_dropout_prob)
        layer_output = bert_modeling.layer_norm(layer_output + attention_output)
        layer_output_shape = bert_modeling.get_shape_list(
            layer_output, expected_rank=3)
        tf.logging.info("layer output shape: %s", layer_output_shape)

        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  return {
      "all_layer_outputs": all_layer_outputs,
      "attention_mask": attention_mask,
      "to_tensors": to_tensors
  }


def add_input_convolutions(x, input_mask, config, pad_up):
  """Adds convolutions to chunk the input."""
  conv_size = config.convolution_size
  # print(f'conv size: {conv_size}')
  if conv_size > 0:
    x = tf.pad(x, paddings=[[0, 0], [conv_size - 1, 0], [0, 0]])
    if input_mask is not None:
      input_mask = tf.pad(input_mask, paddings=[[0, 0], [conv_size - 1, 0]])
    x = tf.layers.conv1d(
        inputs=x,
        filters=config.convolution_filters,
        kernel_size=config.convolution_size,
        strides=1,
        padding="valid")
    if input_mask is not None:
      input_mask = tf.expand_dims(input_mask, -1)
      input_mask = tf.layers.max_pooling1d(
          input_mask,
          pool_size=config.convolution_size,
          strides=1,
          padding="valid")
    x = tf.nn.relu(x)
    x = tf.pad(x, paddings=[[0, 0], [conv_size - 1, 0], [0, 0]])
    if input_mask is not None:
      input_mask = tf.pad(
          input_mask, paddings=[[0, 0], [conv_size - 1, 0], [0, 0]])
    s = bert_modeling.get_shape_list(x, expected_rank=3)[1]
    if pad_up:
      s = (conv_size - (s % conv_size)) % conv_size
      x = tf.pad(x, paddings=[[0, 0], [0, s], [0, 0]])
      if input_mask is not None:
        input_mask = tf.pad(input_mask, paddings=[[0, 0], [0, s], [0, 0]])
    else:
      s = s - (s % conv_size)
      x = x[:, :s, :]
      if input_mask is not None:
        input_mask = input_mask[:, :s, :]
    x = tf.layers.conv1d(
        inputs=x,
        filters=config.convolution_filters * 2,
        kernel_size=config.convolution_size,
        strides=config.convolution_size,
        padding="valid")
    if input_mask is not None:
      input_mask = tf.layers.max_pooling1d(
          input_mask,
          pool_size=config.convolution_size,
          strides=config.convolution_size,
          padding="valid")
      input_mask = tf.squeeze(input_mask, -1)
  return x, input_mask


def add_deconv_layer(x, config):
  """Adds final decoding layer to unchunk the output."""
  if config.deconvolution_size > 0:
    x = tf.layers.conv1d(
        inputs=x,
        filters=config.deconvolution_filters * config.deconvolution_size,
        kernel_size=1,
        strides=1,
        padding="valid")
    s1 = bert_modeling.get_shape_list(x, expected_rank=3)
    x = tf.reshape(
        x,
        [-1, s1[1] * config.deconvolution_size, config.deconvolution_filters])
    # s2 = bert_modeling.get_shape_list(x, expected_rank=3)
    # print(f'Deconvolving {s1} -> {s2}')
  return x


def pad_or_cut(x, length):
  """Fit a tensor to the desired size in the dimension 1."""
  cur = x.shape[1]
  if length > cur:
    x = tf.pad(x, ([0, 0], [0, length - cur], [0, 0]))
  if length < cur:
    x = x[:, :length, :]
  s = x.shape.as_list()
  assert s[1] == length
  return x


def create_causal_feedback(x, conv_size, num_filters, feedback_type):
  """Add autoregressive feedback."""
  if conv_size < 2:
    return None
  sequence_length = x.shape[1]
  if feedback_type == "masked":
    x = tf.layers.conv1d(
        inputs=x,
        filters=num_filters,
        kernel_size=1,
        strides=1,
        padding="valid")
    slices = [
        pad_or_cut(
            tf.repeat(x[:, i::conv_size, :], conv_size, axis=1),
            sequence_length - 1) for i in range(1, conv_size)
    ]
    mask = tf.constant(
        np.tile(1 - np.tri(conv_size, conv_size),
                (sequence_length - 2 + conv_size) //
                conv_size)[:-1, :sequence_length - 1],
        dtype=tf.float32)
    masked = tf.einsum("ij,bijk->bjik", mask, tf.stack(slices, axis=1))
    return tf.reshape(masked, (masked.shape[0], masked.shape[1], -1))
  elif feedback_type == "conv":
    if conv_size < 2:
      return None
    if conv_size > 2:
      x = tf.pad(x, ([0, 0], [0, conv_size - 2], [0, 0]))
    return tf.layers.conv1d(
        inputs=x[:, :-1, :],
        filters=num_filters,
        kernel_size=conv_size - 1,
        strides=1,
        padding="valid")
  else:
    assert feedback_type == "none"
    return None


def add_output_feedback(x, config):
  """Add autoregressive feedback if necessary."""
  if (config.deconvolution_size > 1 and config.convolution_size > 1 and
      config.autoregressive_feedback_filters > 0):
    return create_causal_feedback(x, config.deconvolution_size,
                                  config.autoregressive_feedback_filters,
                                  config.autoregressive_feedback_type)
  return None


class TranslationModel(object):
  """Translation model via Transformers.

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  output_ids = tf.constant([[43, 22, 19], [1, 2, 0]])

  config = TransformerConfig(vocab_size=100, hidden_size=128,
    num_hidden_layers=4, num_attention_heads=8, intermediate_size=512)

  model = TranslationModel(encoder_config=config,
                           decoder_config=config,
                           is_training=True,
                           input_ids=input_ids,
                           output_ids=output_ids)
  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_sequence_output()
  logits = tf.matmul(sequence_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               encoder_config,
               decoder_config,
               is_training,
               input_ids,
               output_ids,
               cached_encoder_output=None,
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for TranslationModel.

    Args:
      encoder_config: `TransformerConfig` instance.
      decoder_config: `TransformerConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, input_seq_length].
      output_ids: int32 Tensor of shape [batch_size, output_seq_length].
      cached_encoder_output: (optional) int32 Tensor of shape [batch_size,
        input_seq_length, hidden_size].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "translator".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    encoder_config = copy.deepcopy(encoder_config)
    decoder_config = copy.deepcopy(decoder_config)
    self.input_mask = tf.cast(tf.math.not_equal(input_ids, 0), dtype=tf.int32)
    if not is_training:
      encoder_config.hidden_dropout_prob = 0.0
      encoder_config.attention_probs_dropout_prob = 0.0
      decoder_config.hidden_dropout_prob = 0.0
      decoder_config.attention_probs_dropout_prob = 0.0

    input_shape = bert_modeling.get_shape_list(input_ids, expected_rank=2)
    output_shape = bert_modeling.get_shape_list(output_ids, expected_rank=2)
    tf.logging.info("input/output token shapes: %s/%s", input_shape,
                    output_shape)
    batch_size = input_shape[0]
    assert batch_size == output_shape[0]
    input_seq_length = input_shape[1]
    output_seq_length = output_shape[1]
    # print(f'output sequence length: {output_seq_length}')
    joined_ids = tf.concat([input_ids, output_ids], axis=1)
    joined_shape = bert_modeling.get_shape_list(joined_ids, expected_rank=2)
    tf.logging.info("joined shape: %s", joined_shape)

    with tf.variable_scope(scope, default_name="translator"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output,
         self.embedding_table) = bert_modeling.embedding_lookup(
             input_ids=joined_ids,
             vocab_size=encoder_config.vocab_size,
             embedding_size=encoder_config.embedding_size,
             initializer_range=encoder_config.initializer_range,
             word_embedding_name="word_embeddings",
             use_one_hot_embeddings=use_one_hot_embeddings)
        self.input_embeddings = self.embedding_output[:, :input_seq_length, :]
        self.output_embeddings = self.embedding_output[:, input_seq_length:, :]
        input_shape = bert_modeling.get_shape_list(
            self.input_embeddings, expected_rank=3)
        output_shape = bert_modeling.get_shape_list(
            self.output_embeddings, expected_rank=3)
        tf.logging.info("input/output embedding shapes: %s/%s", input_shape,
                        output_shape)
        print("input/output embedding shapes: %s/%s" %
              (input_shape, output_shape))
        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.input_embeddings, self.input_mask = add_input_convolutions(
            self.input_embeddings, self.input_mask, encoder_config, pad_up=True)
        mask_shape = bert_modeling.get_shape_list(
            self.input_mask, expected_rank=[2, 3])
        tf.logging.info("input mask shape: %s", mask_shape)
        encoder_shape = bert_modeling.get_shape_list(
            self.input_embeddings, expected_rank=3)
        self.output_feedback = add_output_feedback(self.output_embeddings,
                                                   decoder_config)
        self.output_embeddings, _ = add_input_convolutions(
            self.output_embeddings, None, decoder_config, pad_up=True)
        decoder_shape = bert_modeling.get_shape_list(
            self.output_embeddings, expected_rank=3)
        tf.logging.info("decoder/encoder embedding shapes: %s/%s",
                        decoder_shape, encoder_shape)

        self.input_embeddings = bert_modeling.embedding_postprocessor(
            input_tensor=self.input_embeddings,
            use_position_embeddings=encoder_config.use_position_embeddings,
            position_embedding_name="input_position_embeddings",
            initializer_range=encoder_config.initializer_range,
            max_position_embeddings=encoder_config.max_position_embeddings,
            dropout_prob=encoder_config.hidden_dropout_prob)
        self.output_embeddings = bert_modeling.embedding_postprocessor(
            input_tensor=self.output_embeddings,
            use_position_embeddings=decoder_config.use_position_embeddings,
            position_embedding_name="output_position_embeddings",
            initializer_range=decoder_config.initializer_range,
            max_position_embeddings=decoder_config.max_position_embeddings,
            dropout_prob=decoder_config.hidden_dropout_prob)

      def transformer_from_config(input_tensor, attention_mask, side_tensor,
                                  side_mask, config):
        return transformer_model(
            input_tensor=input_tensor,
            attention_mask=attention_mask,
            side_tensor=side_tensor,
            side_mask=side_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=bert_modeling.get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range)

      with tf.variable_scope("encoder"):
        self.input_attention_mask = (
            bert_modeling.create_attention_mask_from_input_mask(
                self.input_embeddings, self.input_mask))
        input_mask_shape = bert_modeling.get_shape_list(
            self.input_mask, expected_rank=[2, 3])
        input_attention_mask_shape = bert_modeling.get_shape_list(
            self.input_attention_mask, expected_rank=[2, 3])
        tf.logging.info("input/attentaiton mask shapes: %s/%s",
                        input_mask_shape, input_attention_mask_shape)
        tf.logging.info("decoder/encoder embedding shapes: %s/%s",
                        decoder_shape, encoder_shape)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.encoder_layers = transformer_from_config(self.input_embeddings,
                                                      self.input_attention_mask,
                                                      None, None,
                                                      encoder_config)
      self.encoder_output = self.encoder_layers["all_layer_outputs"][-1]

      with tf.variable_scope("decoder"):
        decoder_shape = bert_modeling.get_shape_list(
            self.output_embeddings, expected_rank=3)
        decoder_seq_length = decoder_shape[1]

        # Make attention mask causal (block out attention to future tokens)
        # Create a lower triangular matrix with zero diagonal and 1 below the
        # diagonal.
        num_ones = tf.cast(
            decoder_seq_length * (decoder_seq_length + 1) / 2, dtype=tf.int64)
        causal_mask = tfp.math.fill_triangular(
            tf.ones(num_ones, dtype=tf.float32))
        attention_mask = (
            tf.expand_dims(causal_mask, 0) +
            tf.zeros([batch_size, 1, 1], dtype=tf.float32))
        side_input = self.encoder_layers["all_layer_outputs"][-1]
        decoder_shape = bert_modeling.get_shape_list(
            self.output_embeddings, expected_rank=3)
        mask_shape = bert_modeling.get_shape_list(
            attention_mask, expected_rank=3)
        print(f"decoder input shape: {decoder_shape}/{mask_shape}")
        self.decoder_layers = transformer_from_config(self.output_embeddings,
                                                      attention_mask,
                                                      side_input,
                                                      self.input_mask,
                                                      decoder_config)
      if decoder_config.deconvolution_size > 0:
        assert decoder_config.deconvolution_size == decoder_config.convolution_size
        if decoder_config.deconvolution_filters <= 0:
          decoder_config.deconvolution_filters = decoder_config.convolution_filters
        seq_out = add_deconv_layer(self.decoder_layers["all_layer_outputs"][-1],
                                   decoder_config)
        seq_out = seq_out[:, :(output_seq_length - 1), :]
        print("seq_out:", seq_out.shape.as_list())
        if self.output_feedback is not None:
          print("connecting output feedback")
          feedback_weight = decoder_config.autoregressive_feedback_weight
          num_layers = decoder_config.autoregressive_feedback_layers
          seq_out = tf.concat(
              [1.0 * seq_out, feedback_weight * self.output_feedback], axis=2)
          seq_out = tf.nn.relu(seq_out)
          for _ in range(num_layers):
            seq_out = tf.layers.conv1d(
                inputs=seq_out,
                filters=decoder_config.deconvolution_filters,
                kernel_size=1,
                strides=1,
                padding="valid")
          seq_out = tf.nn.relu(seq_out)
        print("seq_out:", seq_out.shape.as_list())

      self.sequence_output = seq_out
      output_shape = bert_modeling.get_shape_list(
          self.sequence_output, expected_rank=3)
      tf.logging.info("sequence_output shape: %s", output_shape)

      self.eval_sequence_output = None
      if cached_encoder_output is not None:
        with tf.variable_scope("decoder", reuse=True):
          num_ones = tf.cast(
              decoder_seq_length * (decoder_seq_length + 1) / 2, dtype=tf.int64)
          causal_mask = tfp.math.fill_triangular(
              tf.ones(num_ones, dtype=tf.float32))
          attention_mask = (
              tf.expand_dims(causal_mask, 0) +
              tf.zeros([batch_size, 1, 1], dtype=tf.float32))
          self.eval_decoder_layers = transformer_from_config(
              self.output_embeddings, attention_mask, cached_encoder_output,
              self.input_mask, decoder_config)
          self.eval_sequence_output = self.eval_decoder_layers[
              "all_layer_outputs"][-1]

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output


class NextTokenPredictor(object):
  """Next token losses and prediction for sequence modeling."""

  def __init__(self,
               encoder_config,
               decoder_config,
               input_ids,
               output_ids,
               is_training,
               cached_encoder_output=None,
               use_one_hot_embeddings=False):
    """Constructor for NextTokenPredictor.

    Args:
      encoder_config: `TransformerConfig` instance.
      decoder_config: `TransformerConfig` instance.
      input_ids: int32 Tensor of shape [batch_size, input_seq_length].
      output_ids: int32 Tensor of shape [batch_size, output_seq_length].
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      cached_encoder_output: (optional) int32 Tensor of shape [batch_size,
        input_seq_length, hidden_size].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
    """

    self.output_ids = output_ids
    self.output_mask = tf.cast(tf.math.not_equal(self.output_ids, 0), tf.int32)
    self.translator = TranslationModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        is_training=is_training,
        input_ids=input_ids,
        output_ids=self.output_ids,
        cached_encoder_output=cached_encoder_output,
        use_one_hot_embeddings=use_one_hot_embeddings)
    self.targets = output_ids[:, 1:]
    self.encoder_output = self.translator.encoder_output
    self.outputs = self.translator.get_sequence_output()
    self.one_hot_labels = tf.one_hot(self.targets, encoder_config.vocab_size)
    self.logits = tf.compat.v1.layers.dense(
        inputs=self.outputs, units=encoder_config.vocab_size, activation=None)
    self.eval_logits = None
    if cached_encoder_output is not None:
      self.eval_logits = tf.compat.v1.layers.dense(
          inputs=self.translator.eval_sequence_output,
          units=encoder_config.vocab_size,
          activation=None,
          reuse=True)
      print("Using cached logits")
    one_hot_shape = bert_modeling.get_shape_list(
        self.one_hot_labels, expected_rank=3)
    logits_shape = bert_modeling.get_shape_list(self.logits, expected_rank=3)
    print(f"one hot/logits shapes: {one_hot_shape}/{logits_shape}")
    self.per_example_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=self.one_hot_labels, logits=self.logits)
    self.prediction_mask = tf.cast(self.output_mask[:, 1:], tf.float32)
    self.loss = tf.reduce_mean(self.per_example_loss * self.prediction_mask)
    self.predictions = tf.argmax(self.logits, 2)


def model_fn_builder(encoder_config,
                     decoder_config,
                     input_feature,
                     output_feature,
                     init_checkpoint,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     use_tpu,
                     use_one_hot_embeddings,
                     cached_encoder_feature=None,
                     output_dir=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    host_call = None
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features[input_feature]
    output_ids = features[output_feature]
    cached_encoder_output = None
    if cached_encoder_feature:
      cached_encoder_output = features[cached_encoder_feature]
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    predictor = NextTokenPredictor(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        input_ids=input_ids,
        output_ids=output_ids,
        is_training=is_training,
        cached_encoder_output=cached_encoder_output,
        use_one_hot_embeddings=use_one_hot_embeddings)
    total_loss = predictor.loss
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    control_dependencies = []
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = bert_modeling.get_assignment_map_from_checkpoint(
          tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    prediction_mask = predictor.prediction_mask
    if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
      metrics = {}
      running_accuracy, running_accuracy_op = tf.metrics.accuracy(
          labels=predictor.targets,
          predictions=tf.argmax(predictor.logits, -1),
          weights=prediction_mask)
      control_dependencies.append(running_accuracy_op)
      metrics["running_accuracy"] = running_accuracy
      num_predictions = tf.maximum(tf.reduce_sum(prediction_mask), 0.001)
      batch_true_positives = tf.reduce_sum(
          prediction_mask * tf.cast(
              tf.equal(predictor.targets,
                       tf.cast(tf.argmax(predictor.logits, -1),
                               dtype=tf.int32)),
              dtype=tf.float32),
          name="batch_accuracy")
      batch_accuracy = batch_true_positives / num_predictions
      metrics["batch_accuracy"] = batch_accuracy
      metrics["global_step"] = tf.train.get_or_create_global_step()

      def host_call_fn(**kwargs):
        """Call for required on TPUs to produce metrics on the host."""
        with tf2.summary.create_file_writer(
            output_dir, max_queue=10000).as_default():
          gs = kwargs["global_step"][0]
          with tf2.summary.record_if(mode == tf.estimator.ModeKeys.TRAIN):
            for name, metric in kwargs.items():
              if name != "global_step":
                tf2.summary.scalar(
                    f"{name}", tf.reduce_mean(metric, name=f"{name}"), step=gs)
          return tf.summary.all_v2_summary_ops()

      for (name, x) in metrics.items():
        assert x is not None, name
      host_call = (host_call_fn,
                   {name: tf.reshape(x, [1]) for (name, x) in metrics.items()})

      def metric_fn(next_token_logits, next_token_ids, next_token_mean_loss,
                    prediction_mask):
        """Computes the loss and accuracy of the model."""
        next_token_predictions = tf.argmax(
            next_token_logits, axis=-1, output_type=tf.int32)
        next_token_accuracy = tf.metrics.accuracy(
            labels=next_token_ids,
            predictions=next_token_predictions,
            weights=prediction_mask)
        return {
            "next_token_accuracy": next_token_accuracy,
            "next_token_loss": tf.metrics.mean(next_token_mean_loss),
        }

      eval_metrics = (metric_fn, [
          predictor.logits,
          predictor.targets,
          tf.expand_dims(predictor.loss, axis=0),
          predictor.prediction_mask,
      ])
    else:
      eval_metrics = None
    encoder_attention_mask = (
        predictor.translator.encoder_layers["attention_mask"])
    decoder_attention_mask = (
        predictor.translator.decoder_layers["attention_mask"])
    encoder_input_mask = predictor.translator.input_mask
    batch_size, logits_seq_length, _ = (
        bert_modeling.get_shape_list(predictor.logits, expected_rank=3))
    reshaped_logits = tf.reshape(predictor.logits,
                                 (batch_size * logits_seq_length, -1))
    sampling_temperature = tf.constant(1.0)
    sampled = tf.random.categorical(
        sampling_temperature * reshaped_logits, num_samples=1)
    sampled = tf.reshape(sampled, (batch_size, logits_seq_length))

    predictions = {
        "temperature": sampling_temperature,
        "logits": predictor.logits,
        "sampled": sampled,
        "targets": predictor.targets,
        "mask": predictor.prediction_mask,
        "encoder_output": predictor.encoder_output,
        "encoder_input_mask": encoder_input_mask,
        "encoder_attention_mask": encoder_attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
    }

    if cached_encoder_feature:
      reshaped_logits = tf.reshape(predictor.eval_logits,
                                   (batch_size * logits_seq_length, -1))
      sampled = tf.random.categorical(
          sampling_temperature * reshaped_logits, num_samples=1)
      sampled = tf.reshape(sampled, (batch_size, logits_seq_length))
      predictions["eval_logits"] = predictor.eval_logits
      predictions["eval_sampled"] = sampled

    if mode == tf.estimator.ModeKeys.TRAIN:
      with tf.control_dependencies(control_dependencies):
        train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                 num_train_steps,
                                                 num_warmup_steps, use_tpu)
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          predictions=predictions,
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          eval_metrics=eval_metrics,
          host_call=host_call,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          predictions=predictions,
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          host_call=host_call,
          scaffold_fn=scaffold_fn)

    return output_spec

  return model_fn


def input_fn_builder(input_files,
                     input_seq_length,
                     output_seq_length,
                     input_feature,
                     output_feature,
                     is_training,
                     map_fn=None,
                     num_cpu_threads=4,
                     input_format="tfrecord"):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        input_feature: tf.FixedLenFeature([input_seq_length], tf.int64),
        output_feature: tf.FixedLenFeature([output_seq_length], tf.int64)
    }

    assert input_format == "tfrecord"  # default
    dataset_reader = tf.data.TFRecordDataset

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # deterministic=False means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.interleave(
          dataset_reader,
          cycle_length=cycle_length,
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=False)
      d = d.shuffle(buffer_size=100)
    else:
      d = dataset_reader(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` because the TPU requires fixed size dimensions.
    d = d.map(
        lambda record: _decode_record(record, name_to_features),
        num_parallel_calls=num_cpu_threads)
    if map_fn:
      d = d.map(map_fn, num_parallel_calls=num_cpu_threads)
    d = d.batch(batch_size=batch_size, drop_remainder=True)
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(
      record, name_to_features, name="parsing_input_record")

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def hints_preprocess_fn_builder(num_hints,
                                source="output_ids",
                                destination="input_ids"):
  """Creates a `hints_preprocess_fn` preprocessor for adding hints feature."""

  def hints_preprocess_fn(features):
    hints = features[source]
    hints = tf.random.shuffle(hints)[:num_hints]
    features[destination] = hints
    return features

  return hints_preprocess_fn
