import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

y1 = tf.mul(a, b)

# You can have multiple outputs if desired
# its still part of the same graph
# however when running this one (y2), y1 is not considered, TensorFlow is lazy
y2 = tf.add(a, b)


# y1 and y2 are the same type of thing as the placeholders
# you can add them to the graph but no numbers here yet, still just an equation
# its starting to look like a real graph now, file 03 will show you how to see it
y3 = tf.subtract(y1, y2)
# ALGEBRA!
# y3 = y1 - y2
# y3 = (a * b) - y2
# y3 = (a * b) - (a + b)
# y3 = a * b - a - b

with tf.Session() as sess: 
    print("1 * 2 = %d" % sess.run(y1, feed_dict={a: 1, b: 2}))

    print("3 + 3 = %d" % sess.run(y2, feed_dict={a: 3, b: 3}))

    print("(2 * 3) - (2 + 3) = %d" % sess.run(y3, feed_dict={a: 2, b: 3}))
