#!/usr/bin/env python3

import theano
import theano.tensor as T
import numpy as np
import tensorflow as tf


def numpy_combine(t, co, W):
    for coi in co:
        v1, v2 = t[coi[0]], t[coi[1]]
        t[coi[0]] = np.dot(v1, np.dot(W, v2))
    return t[co[-1][0]]


def tensorflow_combine(t, co, W):
    t_buf = tf.Variable(np.zeros((10, 3), dtype=np.float32))
    t_buf = tf.scatter_update(t_buf, list(range(5)), tf.gather(t, list(range(5)))) # tf.range

    def combine_step(ctr, co, t, W):
        coi = tf.gather(co, ctr)
        i1, i2 = tf.unpack(coi, 2)
        v1, v2 = tf.gather(t, i1), tf.gather(t, i2)
        res = tf.matmul(tf.reshape(v2, (1, 3)), tf.reshape(tf.matmul(tf.reshape(v1, (1, 3)), tf.reshape(W, (3, 3 * 3))), (3, 3)))
        res = tf.reshape(res, (3,))
        return tf.add(ctr, 1), co, tf.scatter_update(t, i1, res), W

    r = tf.python.control_flow_ops.While(
            lambda ctr, co, t, W: tf.python.math_ops.less(ctr, 4),
            combine_step,
            [tf.constant(0), co, t_buf, W]
            )
    return r[2]



def test_numpy():
    tree = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.float32)
    combination_order = [[1, 2], [0, 1], [3, 4], [0, 3]]
    W_init = np.ones((3, 3, 3), dtype=np.float32)

    numpy_result = numpy_combine(tree, combination_order, W_init)
    print("numpy result: {}".format(numpy_result))


def test_tensorflow():
    tree = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.float32)
    combination_order = [[1, 2], [0, 1], [3, 4], [0, 3]]
    W_init = np.ones((3, 3, 3), dtype=np.float32)

    with tf.Session() as sess:
        Tree = tf.placeholder(tf.float32)
        CombOrd = tf.placeholder(tf.int32)

        W = tf.Variable(W_init, name='W')

        r = tensorflow_combine(Tree, CombOrd, W)

        sess.run(tf.initialize_all_variables())

        print(sess.run(r, feed_dict={Tree: tree, CombOrd: combination_order}))


def main():
    test_numpy()
    test_tensorflow()

if __name__ == '__main__':
    main()
