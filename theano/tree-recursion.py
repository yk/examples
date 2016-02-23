#!/usr/bin/env python3

import theano
import theano.tensor as T
import numpy as np


def numpy_combine(t, co, W):
    for coi in co:
        v1, v2 = t[coi[0]], t[coi[1]]
        t[coi[0]] = np.dot(v1, np.dot(W, v2))
    return t[co[-1][0]]


def theano_combine(t, co, W):

    def combine_step(coi, t, W):
        v1, v2 = t[coi[0]], t[coi[1]]
        res = T.dot(v1, T.dot(W, v2))
        t_new = T.set_subtensor(t[coi[0], :], res)
        return t_new
    outputs, updates = theano.scan(combine_step, sequences=[co], outputs_info=t, non_sequences=[W])
    result = outputs[-1][co[-1][0]]
    return result, updates


def test_numpy():
    tree = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.float32)
    combination_order = [[1, 2], [0, 1], [3, 4], [0, 3]]
    W_init = np.ones((3, 3, 3), dtype=np.float32)

    numpy_result = numpy_combine(tree, combination_order, W_init)
    print("numpy result: {}".format(numpy_result))


def test_theano():
    tree = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=np.float32)
    combination_order = [[1, 2], [0, 1], [3, 4], [0, 3]]
    W_init = np.ones((3, 3, 3), dtype=np.float32)

    Tree = T.matrix('tree')
    CombOrd = T.imatrix('combination_order')

    W = theano.shared(W_init, 'W')

    result, updates = theano_combine(Tree, CombOrd, W)

    theano_comb_func = theano.function([Tree, CombOrd], result, updates=updates)

    theano_result = theano_comb_func(tree, combination_order)
    print("theano result: {}".format(theano_result))


def main():
    test_numpy()
    test_theano()

if __name__ == '__main__':
    main()
