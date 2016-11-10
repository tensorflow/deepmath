"""Collection of benchmark models for classifying proof steps with
conditioning on the conjecture.
"""
from keras import layers
from keras.models import Model


def cnn_2x_siamese(voc_size, max_len, dropout=0.5):
    """Two siamese branches, each embedding a statement.
    Binary classifier on top.
    Args:
      voc_size: size of the vocabulary for the input statements.
      max_len: maximum length for the input statements.
    Returns:
      A Keras model instance.
    """
    pivot_input = layers.Input(shape=(max_len,), dtype='int32')
    statement_input = layers.Input(shape=(max_len,), dtype='int32')

    x = layers.Embedding(
        output_dim=256,
        input_dim=voc_size,
        input_length=max_len)(pivot_input)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    embedded_pivot = layers.GlobalMaxPooling1D()(x)

    encoder_model = Model(pivot_input, embedded_pivot)
    embedded_statement = encoder_model(statement_input)

    concat = layers.merge([embedded_pivot, embedded_statement], mode='concat')
    x = layers.Dense(256, activation='relu')(concat)
    x = layers.Dropout(dropout)(x)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    model = Model([pivot_input, statement_input], prediction)
    return model


def cnn_2x_lstm_siamese(voc_size, max_len, dropout=0.5):
    """Two siamese branches, each embedding a statement.
    Binary classifier on top.
    Args:
      voc_size: size of the vocabulary for the input statements.
      max_len: maximum length for the input statements.
    Returns:
      A Keras model instance.
    """
    pivot_input = layers.Input(shape=(max_len,), dtype='int32')
    statement_input = layers.Input(shape=(max_len,), dtype='int32')

    x = layers.Embedding(
        output_dim=256,
        input_dim=voc_size,
        input_length=max_len)(pivot_input)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Convolution1D(256, 7, activation='relu')(x)
    x = layers.MaxPooling1D(5)(x)
    embedded_pivot = layers.LSTM(256)(x)

    encoder_model = Model(pivot_input, embedded_pivot)
    embedded_statement = encoder_model(statement_input)

    concat = layers.merge([embedded_pivot, embedded_statement], mode='concat')
    x = layers.Dense(256, activation='relu')(concat)
    x = layers.Dropout(dropout)(x)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    model = Model([pivot_input, statement_input], prediction)
    return model


def embedding_logreg_siamese(voc_size, max_len, dropout=0.5):
    """Two siamese branches, each embedding a statement.
    Binary classifier on top.
    Args:
      voc_size: size of the vocabulary for the input statements.
      max_len: maximum length for the input statements.
    Returns:
      A Keras model instance.
    """
    pivot_input = layers.Input(shape=(max_len,), dtype='int32')
    statement_input = layers.Input(shape=(max_len,), dtype='int32')

    x = layers.Embedding(
        output_dim=256,
        input_dim=voc_size,
        input_length=max_len)(pivot_input)
    x = layers.Activation('relu')(x)
    embedded_pivot = layers.Flatten()(x)

    encoder_model = Model(pivot_input, embedded_pivot)
    embedded_statement = encoder_model(statement_input)

    concat = layers.merge([embedded_pivot, embedded_statement], mode='concat')
    x = layers.Dropout(dropout)(concat)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    model = Model([pivot_input, statement_input], prediction)
    return model


# Contains both the model definition function and the type of encoding needed.
MODELS = {
    'cnn_2x_siamese': (cnn_2x_siamese, 'integer'),
    'cnn_2x_lstm_siamese': (cnn_2x_lstm_siamese, 'integer'),
    'embedding_logreg_siamese': (embedding_logreg_siamese, 'integer'),
}
