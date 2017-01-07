import tensorflow as tf
from datetime import datetime

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

y1 = a * b
y2 = a + b
y3 = y1 - y2

# TensorBoard is a nice UI to show you what is going on
# It's a separate program which reads a file
# It's part of a typical TensorFlow install, so you should already have it

# A unique folder name per run makes things manageable,
# otherwise TensorBoard will try to shove all results into one view and be terribly confusing
summaryDirectory = "./summary/" + datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

# This will write out the summary files
summaryWriter = tf.summary.FileWriter(summaryDirectory)

# Add the graph to the summary
summaryWriter.add_graph(tf.get_default_graph())
# if you wonder why it says default graph it is because 
# you can have multiple separate graphs
# but you don't need to worry about that now

with tf.Session() as sess: 
    print("1 * 2 = %d" % sess.run(y1, feed_dict={a: 1, b: 2}))

    print("3 + 3 = %d" % sess.run(y2, feed_dict={a: 3, b: 3}))

    print("(2 * 3) - (2 + 3) = %d" % sess.run(y3, feed_dict={a: 2, b: 3}))


    # Yes, do these things
    print("Run: tensorboard --logdir=./summary" \
        "\nThen open http://localhost:6006/ into your web browser")

    # To see the graph, click the Graphs bit at the top

    # You can scroll to zoom in, drag the mouse to pan
    # click on the nodes and it will tell you all sorts of information
    # it will not show you the data in the graphs though

    # Important:
    # If you run these multiple times you will need to be carful as to which one you are viewing
    # in the Graph tab there is a Run dropdown box to change your selection
    # if in doubt, stop tensorboard (ctrl+c), delete the summary directory and start again
