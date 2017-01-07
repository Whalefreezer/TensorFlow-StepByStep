import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# The code in this file is exactly identical to the previous one
# just using a slightly different syntax

# TensorFlow does operator overloading in python
y1 = a * b

y2 = a + b

y3 = y1 - y2

# Caution:
# * and / might not do what you expect when dealing with matrices (tensors)
# when in doubt use tf.matmul(a, b), tf.mul(a, b) etc. (these 2 are different!)
# https://www.tensorflow.org/api_docs/python/math_ops/

with tf.Session() as sess: 
    print("1 * 2 = %d" % sess.run(y1, feed_dict={a: 1, b: 2}))

    print("3 + 3 = %d" % sess.run(y2, feed_dict={a: 3, b: 3}))

    print("(2 * 3) - (2 + 3) = %d" % sess.run(y3, feed_dict={a: 2, b: 3}))
