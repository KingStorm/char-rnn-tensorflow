from __future__ import print_function

import argparse
import os
from six.moves import cPickle
from six import text_type
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
print("visible cuda devices: %s" % os.environ["CUDA_VISIBLE_DEVICES"])

import tensorflow as tf
from model import Model

_VERBOSE_ = False


class API():
    def __init__(self, save_dir):
        with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
            saved_args = cPickle.load(f)
        with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
            chars, self.vocab = cPickle.load(f)
        self.model = Model(saved_args, training=False)

        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore %s" % ckpt)
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def predict(self, text, verbose=_VERBOSE_):
        state = self.sess.run(self.model.cell.zero_state(1, tf.float32))
        score = 0
        text = text.split()

        # It's like start tag.
        x = np.zeros((1, 1))
        x[0, 0] = self.vocab['_BOUND_']
        feed = {self.model.input_data: x, self.model.initial_state: state}
        [probs, state] = self.sess.run([self.model.probs, self.model.final_state], feed)

        p = probs[0]
        score += np.log(p[self.vocab[text[0]]] + 1e-10)

        # Loop over top n-1 chars.
        for ind in np.arange(len(text)-1):
            char = text[ind]
            char_next = text[ind+1]

            x = np.zeros((1, 1))
            x[0, 0] = self.vocab[char]
            feed = {self.model.input_data: x, self.model.initial_state: state}
            [probs, state] = self.sess.run([self.model.probs, self.model.final_state], feed)

            p = probs[0]
            score += np.log(p[self.vocab[char_next]] + 1e-10)
            if verbose:
                print("%s, score=%f, acc_log_score=%f" % (
                    char_next, p[self.vocab[char_next]], score))

        return score


if __name__ == "__main__":
    api = API("./save")
    text = "测 试 测 试 测 试 ▁PAP CC"
    score = api.predict(text)
    print(text)
    print(score)
