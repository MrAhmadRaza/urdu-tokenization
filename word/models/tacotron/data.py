"""Dataset"""

import pickle
import numpy as np
import tensorflow as tf
import config
from urdu_alphabet import URDU_ALPHABET_COMPLETE


def get_vocab_mapping():
    """
    Loading the Urdu character mapping
    Returns:
        tuple
    """

    vocab = sorted(URDU_ALPHABET_COMPLETE)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    return word2idx, idx2word


def get_data(mode: str = "train"):
    """
    Load the urdu sentences list
    Args:
        mode: train / test data

    Returns:

    """

    with open(config.DATA_FILE, 'rb') as f:
        sentences = pickle.load(f)

    word2idx, idx2word = get_vocab_mapping()

    xs, ys = [], []
    for index, sentence in enumerate(sentences):

        sentence = ' '.join(sentence)

        if config.SENTENCE_MIN_LENGTH <= len(sentence) <= \
                config.SENTENCE_MAX_LENGTH:

            x, y = [], []
            for word in sentence.split():
                for char in word:

                    y.append(0)
                    x.append(word2idx[char])

                y[-1] = 1  # space for end of a word
            y[-1] = 0  # no space for end of sentence

            xs.append(x + [0] * (config.SENTENCE_MAX_LENGTH - len(x)))
            ys.append(y + [0] * (config.SENTENCE_MAX_LENGTH - len(x)))

    x_ = np.array(xs, np.int32)
    y_ = np.array(ys, np.int32)

    if mode == "train":

        x_, y_ = x_[: int(len(x_) * .8)], \
                 y_[: int(len(y_) * .8)]

    elif mode == "val":
        x_, y_ = x_[int(len(x_) * .8): -int(len(x_) * .1)] \
            , y_[int(len(x_) * .8): -int(len(x_) * .1)]
    else:
        x_, y_ = x_[-int(len(x_) * .1):], y_[-int(len(x_) * .1):]

    return x_, y_


def get_batch_data():
    """
    Load the Batch data
    Returns:
        tuple
    """

    x_, y_ = get_data()

    num_batch = len(x_) // config.BATCH_SIZE

    x_ = tf.convert_to_tensor(x_, tf.int32)
    y_ = tf.convert_to_tensor(y_, tf.int32)

    input_queues = tf.train.slice_input_producer([x_, y_])

    x, y = tf.train.batch(input_queues,
                          num_threads=8,
                          batch_size=config.BATCH_SIZE,
                          capacity=config.BATCH_SIZE * 64,
                          allow_smaller_final_batch=False)

    return x, y, num_batch
