"""Tacotron Modules"""
from __future__ import print_function
import tensorflow as tf


def embedding(inputs, vocab_size, num_units, zero_pad=True, scale=True,
              scope="embedding",
              reuse=None):
    """

    Args:
        inputs:
        vocab_size:
        num_units:
        zero_pad:
        scale:
        scope:
        reuse:

    Returns:

    """

    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


def normalize(inputs, type="bn", decay=.999, epsilon=1e-8, is_training=True,
              activation_fn=None,
              reuse=None,
              scope="normalize"):
    """

    Args:
        inputs:
        type:
        decay:
        epsilon:
        is_training:
        activation_fn:
        reuse:
        scope:

    Returns:

    """

    if type == "bn":
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims

        if inputs_rank in [2, 3, 4]:
            if inputs_rank == 2:
                inputs = tf.expand_dims(inputs, axis=1)
                inputs = tf.expand_dims(inputs, axis=2)
            elif inputs_rank == 3:
                inputs = tf.expand_dims(inputs, axis=1)

            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                   decay=decay,
                                                   center=True,
                                                   scale=True,
                                                   activation_fn=activation_fn,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   scope=scope,
                                                   zero_debias_moving_mean=True,
                                                   fused=True,
                                                   reuse=reuse)
            # restore original shape
            if inputs_rank == 2:
                outputs = tf.squeeze(outputs, axis=[1, 2])
            elif inputs_rank == 3:
                outputs = tf.squeeze(outputs, axis=1)
        else:  # fallback to naive batch norm
            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                                   decay=decay,
                                                   center=True,
                                                   scale=True,
                                                   activation_fn=activation_fn,
                                                   updates_collections=None,
                                                   is_training=is_training,
                                                   scope=scope,
                                                   reuse=reuse,
                                                   fused=False)
    elif type in ("ln", "ins"):
        reduction_axis = -1 if type == "ln" else 1
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [reduction_axis],
                                           keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta
    else:
        outputs = inputs

    if activation_fn:
        outputs = activation_fn(outputs)

    return outputs


def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           padding="SAME",
           use_bias=False,
           activation_fn=None,
           scope="conv1d",
           reuse=None):
    """

    Args:
        inputs:
        filters:
        size:
        rate:
        padding:
        use_bias:
        activation_fn:
        scope:
        reuse:

    Returns:

    """

    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list[-1]

        params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding,
                  "activation": activation_fn,
                  "use_bias": use_bias, "reuse": reuse}

        outputs = tf.layers.conv1d(**params)
    return outputs


def conv1d_banks(inputs, K=16, num_units=None, norm_type=None,
                 is_training=True, scope="conv1d_banks", reuse=None):
    """

    Args:
        inputs:
        K:
        num_units:
        norm_type:
        is_training:
        scope:
        reuse:

    Returns:

    """

    if num_units is None:
        num_units = inputs.get_shape()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv1d(inputs, num_units, 1)  # k=1
        for k in range(2, K + 1):  # k = 2...K
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, num_units, k)
                outputs = tf.concat((outputs, output), -1)
        outputs = normalize(outputs, type=norm_type, is_training=is_training,
                            activation_fn=tf.nn.relu)

    return outputs  # (N, T, Hp.embed_size//2*K)


def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
    """

    Args:
        inputs:
        num_units:
        bidirection:
        scope:
        reuse:

    Returns:

    """

    if num_units is None:
        num_units = inputs.get_shape()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]

        cell = tf.contrib.rnn.GRUCell(num_units)
        if bidirection:
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs,
                                                         dtype=tf.float32)
            return tf.concat(outputs, 2)
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            return outputs


def prenet(inputs, num_units=None, dropout_rate=0, is_training=True,
           scope="prenet", reuse=None):
    """

    Args:
        inputs:
        num_units:
        dropout_rate:
        is_training:
        scope:
        reuse:

    Returns:

    """

    if num_units is None:
        num_units = [inputs.get_shape()[-1], inputs.get_shape()[-1]]

    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, units=num_units[0],
                                  activation=tf.nn.relu, name="dense1")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate,
                                    training=is_training, name="dropout1")
        outputs = tf.layers.dense(outputs, units=num_units[1],
                                  activation=tf.nn.relu, name="dense2")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate,
                                    training=is_training, name="dropout2")

    return outputs  # (N, T, num_units[1])


def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    """

    Args:
        inputs:
        num_units:
        scope:
        reuse:

    Returns:

    """
    if num_units is None:
        num_units = inputs.get_shape()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu,
                            name="H")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                            name="T")
        C = 1. - T
        outputs = H * T + inputs * C

    return outputs
