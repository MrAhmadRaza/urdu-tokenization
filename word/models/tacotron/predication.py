"""Generate Predication """
from __future__ import print_function, unicode_literals

import os
import config
import tensorflow as tf
from train import Graph
from data import get_vocab_mapping, get_data, get_batch_data


def eval():
    """
    Eval the result
    Returns:

    """
    g = Graph(is_training=False)
    print("Graph loaded")

    X, Y = get_data(mode="test")
    char2idx, idx2char = get_vocab_mapping()

    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(
                config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(config.LOG_DIR))
            print("Restored!")

            # Get model
            mname = open(config.LOG_DIR + '/checkpoint', 'r').read().split('"')[
                1]  # model name

            # Inference
            if not os.path.exists(config.SAVE_DIR):
                os.mkdir(config.SAVE_DIR)
            with open("{}/{}".format(config.SAVE_DIR, mname), 'w') as fout:
                results = []
                baseline_results = []
                for step in range(len(X) // config.BATCH_SIZE):
                    x = X[step * config.BATCH_SIZE: (step + 1) * config.BATCH_SIZE]
                    y = Y[step * config.BATCH_SIZE: (step + 1) * config.BATCH_SIZE]

                    # predict characters
                    preds = sess.run(g.preds, {g.x: x})

                    for xx, yy, pp in zip(x, y, preds):  # sentence-wise
                        expected = ''
                        got = ''
                        for xxx, yyy, ppp in zip(xx, yy, pp):  # character-wise
                            if xxx == 0:
                                break
                            else:
                                got += idx2char.get(xxx, "*")
                                expected += idx2char.get(xxx, "*")
                            if ppp == 1:
                                got += "-"
                            if yyy == 1:
                                expected += "-"

                            # prediction results
                            if ppp == yyy:
                                results.append(1)
                            else:
                                results.append(0)

                            # baseline results
                            if yyy == 0:  # no space
                                baseline_results.append(1)
                            else:
                                baseline_results.append(0)

                        fout.write("Raw Sentence: " + expected + "\n")
                        fout.write("Predication Sentence: " + got + "\n\n")
                fout.write(
                    "Final Accuracy = %d/%d=%.4f\n" % (
                        sum(results), len(results),
                        float(sum(results)) / len(results)))
                fout.write(
                    "Baseline Accuracy = %d/%d=%.4f" % (
                        sum(baseline_results), len(baseline_results),
                        float(sum(baseline_results)) / len(baseline_results)))


if __name__ == '__main__':
    eval()
    print("Done")
