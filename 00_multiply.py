import tensorflow as tf

# Placeholders:
# Trust me bro, I don't have a value for your graph yet,
# but ill give you one when you run.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Does not do any math here,
# just sets the idea that y = a * b
y = tf.mul(a, b)
# the idea that these values (a, b) flow into y is like a graph
# most stuff refer to it as The Graph

# you need to make a session for some reason
# also for some reason it's a good idea to delay it until you need it
with tf.Session() as sess: 

    # Runs the command "y"
    # MUST enter all placeholder values now, keyname is the variable name
    yVal = sess.run(y, feed_dict={a: 1, b: 2})
    # its very easy to get y and yVal mixed up
    # y is the concept, equation, a node on the graph, (y = a * b)
    # yVal is after you've substituted the variables in and is just a number (2.0)

    print("1 * 2 = %d" % yVal)

    # you can evaluate this graph (equation) as many times as you want 
    print("3 * 3 = %d" % sess.run(y, feed_dict={a: 3, b: 3}))
