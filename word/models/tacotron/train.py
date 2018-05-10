"""Training the network"""
from __future__ import print_function, unicode_literals

import tensorflow as tf

import config
from data import get_vocab_mapping, get_data, get_batch_data
from modules import *
from tqdm import tqdm


class Graph:
    """
    Main graph
    """

    def __init__(self, is_training=True):

        self.graph = tf.Graph()
        with self.graph.as_default():

            self.x, self.y, self.num_batch = get_batch_data()

            char2idx, idx2char = get_vocab_mapping()

            enc = embedding(self.x,
                            vocab_size=len(char2idx),
                            num_units=config.HIDDEN_UNITS,
                            scale=False,
                            scope="enc_embed")


            prenet_out = prenet(enc,
                                num_units=[config.HIDDEN_UNITS,
                                           config.HIDDEN_UNITS // 2],
                                dropout_rate=config.DROPOUT_RATE,
                                is_training=is_training)  # (N, T, E/2)


            enc = conv1d_banks(prenet_out,
                               K=config.ENCODER_NUM_BANKS,
                               num_units=config.HIDDEN_UNITS // 2,
                               norm_type="ins",
                               is_training=is_training)  # (N, T, K * E / 2)


            enc = tf.layers.max_pooling1d(enc, 2, 1,
                                          padding="same")  # (N, T, K * E / 2)


            enc = conv1d(enc, config.HIDDEN_UNITS // 2, 3,
                         scope="conv1d_1")  # (N, T, E/2)
            enc = normalize(enc, type="ins", is_training=is_training,
                            activation_fn=tf.nn.relu)
            enc = conv1d(enc, config.HIDDEN_UNITS // 2, 3,
                         scope="conv1d_2")  # (N, T, E/2)
            enc += prenet_out  # (N, T, E/2) # residual connections


            for i in range(config.NUM_HIGHWAYNET_BLOCKS):
                enc = highwaynet(enc, num_units=config.HIDDEN_UNITS // 2,
                                 scope='highwaynet_{}'.format(
                                     i))  # (N, T, E/2)


            enc = gru(enc, config.HIDDEN_UNITS // 2, True)  # (N, T, E)

            # Final linear projection
            self.logits = tf.layers.dense(enc,
                                          2)  # 0 for non-space, 1 for space

            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
            self.istarget = tf.to_float(tf.not_equal(self.x, 0))  # masking
            self.num_hits = tf.reduce_sum(
                tf.to_float(tf.equal(self.preds, self.y)) * self.istarget)
            self.num_targets = tf.reduce_sum(self.istarget)
            self.acc = self.num_hits / self.num_targets

            if is_training:
                # Loss
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.y)
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (
                    tf.reduce_sum(self.istarget))

                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step',
                                               trainable=False)
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=config.LEARNING_RATE,
                    beta1=0.9, beta2=0.98,
                    epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss,
                                                        global_step=self.global_step)

                # # Summary
                # tf.summary.scalar('mean_loss', self.mean_loss)
                # tf.summary.merge_all()


if __name__ == '__main__':
    # Construct graph
    g = Graph()
    print("Graph loaded")

    char2idx, idx2char = get_vocab_mapping()
    with g.graph.as_default():
        # For validation
        X_val, Y_val = get_data(mode="val")
        num_batch = len(X_val) // config.BATCH_SIZE

        # Start session
        sv = tf.train.Supervisor(graph=g.graph,
                                 logdir=config.LOG_DIR,
                                 save_model_secs=0)
        with sv.managed_session() as sess:
            for epoch in range(1, config.NUM_EPOCHS + 1):
                if sv.should_stop():
                    break
                for step in tqdm(range(g.num_batch), total=g.num_batch,
                                 ncols=70, leave=False, unit='b'):
                    sess.run(g.train_op)

                    # logging
                    if step % 100 == 0:
                        gs, mean_loss = sess.run([g.global_step, g.mean_loss])
                        print(
                            "\nAfter global steps %d, the training loss is %.2f" % (
                                gs, mean_loss))

                # Save
                gs = sess.run(g.global_step)
                sv.saver.save(sess,
                              config.LOG_DIR + '/model_epoch_%02d_gs_%d' % (
                                  epoch, gs))

                # Validation check
                total_hits, total_targets = 0, 0
                for step in tqdm(range(num_batch), total=num_batch, ncols=70,
                                 leave=False, unit='b'):
                    x = X_val[step * config.BATCH_SIZE:(
                                                               step + 1) * config.BATCH_SIZE]
                    y = Y_val[step * config.BATCH_SIZE:(
                                                               step + 1) * config.BATCH_SIZE]
                    num_hits, num_targets = sess.run(
                        [g.num_hits, g.num_targets], {g.x: x, g.y: y})
                    total_hits += num_hits
                    total_targets += num_targets
                print(
                    "\nAfter epoch %d, the validation accuracy is %d/%d=%.2f" % (
                        epoch, total_hits, total_targets,
                        total_hits / total_targets))

    print("Done")
