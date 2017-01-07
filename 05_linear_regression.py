import tensorflow as tf
import random

# Whooo, time for some actual gradient descent machine learning stuff!
# well sorta, not really... this still can be done with a (fancy) calculator

# In this example we are going to take a bunch of 2d points (x, y)
# and try to make a equation that fits it ok-ish
# we are going to use y = mx + c
# goal is to find out m and c

# For this to work it needs a bunch of examples to "learn" from:
trainingData = []
for i in range(100):
    xVal = i / 100.0

    yVal = 2 * xVal + 5
    # yes, I know you know m = 2 and c = 5 now, but the computer doesn't, so shhhh...

    # Add some noise to make it a little bit interesting
    noise = random.normalvariate(0, 0.3)
    trainingData.append((xVal, yVal + noise))

# Values to be provided at run time
# can be input, values to compare the output to or whatever
x = tf.placeholder(tf.float32)
yCorrect = tf.placeholder(tf.float32)

# This is new
# the bad news is that you or the computer don't know the values of these
# the good news is that it's the computers problem, not yours
# Variables are what the computer will change to solve the problem
m = tf.Variable(0.0)
c = tf.Variable(0.0)

# Bam! here is the model
y = m * x + c

# We need to tell TensorFlow how good of a job it is doing
# this is called cost, loss, error etc. mostly mean the same thing
# the goal is to get rid of it (minimize)

# you can have any cost function you like (some are better than others)
# however it must make sense

# eg if you cost function is "correct - test"
# when everything is fine then "correct = test", "5 = 5", "cost = 5 - 5 = 0"
# however when "test = 8" then "cost = 5 - 8 = -3" which is less than before so it's a better solution right!?
# easy answer to this is to put it inside some abs, |correct - test|
# it will be 0 if its right and more than that if its wrong, and the more wrong, the more bigger, excellent :D 

# However for reasons, math guys like squaring it instead, same results though
cost = tf.square(yCorrect - y)

# Here is the magic, the reason why we are bothering with TensorFlow
# this will use black magic to move Variables to make cost 0
trainOp = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    # If you have variables you need to do this for some reason
    tf.global_variables_initializer().run()

    # Go over all the data 50 times, called epoch number
    for i in range(50):
        for xVal, yVal in trainingData:
            # trainOp -> black magic -> cost -> yCorrect, y -> m, x, c
            sess.run(trainOp, feed_dict={x: xVal, yCorrect: yVal})

    # To make it spit out m and c we need to run them
    # don't need feed_dict as they don't depend on any placeholders
    mVal, cVal = sess.run([m, c])
    print("y = %.1fx + %.1f" % (mVal, cVal))
