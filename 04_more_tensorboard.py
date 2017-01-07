import tensorflow as tf
from datetime import datetime

# You can give most things names
# this only has an effect on TensorBoard
# these are named oddly to show it has no relation to the name that the feed_dict needs
a = tf.placeholder(tf.float32, name="AAA")
b = tf.placeholder(tf.float32, name="BBB")

# you can add names to operations
# but only using the old syntax :(
y1 = tf.mul(a, b, name="y1")

# You can also add scopes, anything inside it goes in a box
# these boxes will be collapsed by default, makes it easier to view
# they can be expanded by double clicking on them
with tf.name_scope('360noscope') as scope:
    y2 = tf.add(a, b, name="y2")
    y3 = tf.subtract(y1, y2, name="y3")

summaryDirectory = "./summary/" + datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
summaryWriter = tf.summary.FileWriter(summaryDirectory)
summaryWriter.add_graph(tf.get_default_graph())

with tf.Session() as sess: 
    print("1 * 2 = %d" % sess.run(y1, feed_dict={a: 1, b: 2}))

    print("3 + 3 = %d" % sess.run(y2, feed_dict={a: 3, b: 3}))

    print("(2 * 3) - (2 + 3) = %d" % sess.run(y3, feed_dict={a: 2, b: 3}))

    print("Run: tensorboard --logdir=./summary" \
        "\nThen open http://localhost:6006/ into your web browser")
