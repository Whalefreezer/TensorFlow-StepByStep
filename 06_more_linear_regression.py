import tensorflow as tf
from datetime import datetime
import numpy as np

# This does exactly the same as before
# but since numpy is such good friends with TensorFlow its a good idea to learn it
xPoints = np.linspace(0, 1, 100)
yPoints = 2 * xPoints + np.random.standard_normal(xPoints.shape) * 0.3 + 5
trainingData = np.stack((xPoints, yPoints)).T

x = tf.placeholder(tf.float32)
yCorrect = tf.placeholder(tf.float32)

m = tf.Variable(0.0)
c = tf.Variable(0.0)

y = m * x + c

cost = tf.square(yCorrect - y)

trainOp = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# Add in TensorBoard stuff
summaryDirectory = "./summary/" + datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
summaryWriter = tf.summary.FileWriter(summaryDirectory)
summaryWriter.add_graph(tf.get_default_graph())

# This is new, it will make a graph in tensorboard showing how the loss (cost) decreased
tf.summary.scalar("loss", cost)

# This takes all of the summary stuff declared before it (in this case just the line above)
# and makes an operation that needs to be run get data
mergedSummaryOp = tf.summary.merge_all()

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    step = 0
    for i in range(50):
        for xVal, yVal in trainingData:
            _, summary = sess.run([trainOp, mergedSummaryOp], feed_dict={x: xVal, yCorrect: yVal})
            step += 1
            # The summary for each step needs to be written out
            summaryWriter.add_summary(summary, step)

    mVal, cVal = sess.run([m, c])
    print("y = %.1fx + %.1f" % (mVal, cVal))

    print("Run: tensorboard --logdir=./summary" \
        "\nThen open http://localhost:6006/ into your web browser")
