#!/usr/bin/env python3

import theano
import theano.tensor as T

y = T.ivector('y')

y_oh = T.extra_ops.to_one_hot(y, 4)

f = theano.function([y], y_oh)

print(f([0, 2, 3]))

# prints:
# [[ 1.  0.  0.  0.]
#  [ 0.  0.  1.  0.]
#  [ 0.  0.  0.  1.]]
