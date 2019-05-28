# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#
# batch_size = 100
# n_batch = mnist.train.num_examples // batch_size
#
# x = tf.placeholder(tf.float32,[None,784])
# y = tf.placeholder(tf.float32,[None,10])
# W = tf.Variable(tf.zeros([784,10]))
# B = tf.Variable(tf.zeros([10]))
# prediction = tf.nn.softmax(tf.matmul(W,x) + B)
# loss = tf.reduce_mean(tf.square(y - prediction))
# train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
# correction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
# accuracy = tf.reduce_mean(tf.assign(correction),tf.float32)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer)
#     for epoch in  range(21):
#         for batch in range(n_batch):
#             batch_xs,batch_ys = mnist.train.next_batch(batch_size)
#             sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
#         acc =sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
#         print("Iter:"+str(epoch) + ",Testing Accuracy:" + str(acc))
#



import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

batch_size = 200
n_batch = mnist.train.num_examples // batch_size
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W) + b)

#loss = tf.reduce_mean(tf.square(y - prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    # print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
    # saver.restore(sess,'net/first.ckpt')
    # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    for epoch in range(200):
        for batch in range(batch_size):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter "+str(epoch)+",Test Accuracy " + str(acc))
    #saver.save(sess,'net/first.ckpt')