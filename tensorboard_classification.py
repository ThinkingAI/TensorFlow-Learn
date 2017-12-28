# -*- coding: utf-8 -*-
# -*- author: knock -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
MNIST数据集，每一张图片包含28X28个像素点
'''
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
print(mnist.train.images.shape)  #(55000, 784) ->X: 55000 training set; Y: 28*28展开后的one_hot
print(mnist.train.labels.shape)  #(55000, 10)  ->X: 55000             ; Y: 0-9 one_hot 形式
print(mnist.test.images.shape)   #(10000, 784)->10000 test set
print(mnist.test.labels.shape)   #(10000, 10)

#添加网络层
def add_layer(inputs,in_size,out_size,activation_fucntion=None,name='layer'):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
        if activation_fucntion is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_fucntion(Wx_plus_b)
        return  outputs

# 计算验证集的准确率
def compute_accuracy(validate_xs,validate_ys):
    global prediction
    y_preiction = sess.run(prediction, feed_dict={xs: validate_xs})
    #argmax 计算出其最大值所在的索引位置，如果是0,1 one_hot形式，也就是返回1所在索引值
    #预测索引值和真实索引值是否一致，返回bool数组结果
    correct_prediction = tf.equal(tf.argmax(y_preiction, 1), tf.argmax(validate_ys, 1))
    # 给出一组bool值，如[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: validate_xs, ys: validate_ys})
    return result
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784], name='input_x') # None means: 任意数字样本条数，后续单批批量数
    ys = tf.placeholder(tf.float32, [None, 10], name='input_y')  # 实际结果，用于和预测结果对比

with tf.variable_scope('Net'):
    # add hidden layer
    hidden1 = add_layer(xs, 784, 784*2, activation_fucntion=tf.nn.softmax,name='hidden1')
    tf.summary.histogram('hidden1',hidden1) #作用于层监控数据

    #add output layer
    prediction = add_layer(hidden1,784*2,10,activation_fucntion=tf.nn.softmax,name='output')
    tf.summary.histogram('prediction',prediction)
#cross entropy 交叉熵公式计算
#[1]在第二维度上面求均值交叉熵
with tf.name_scope('loss'):
    mean_cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices = 1),name='loss')
    tf.summary.scalar('loss', mean_cross_entropy)     # 作用于标量，将loss添加到Tensorboard标量
# 训练operation，利用优化器优化cost
with tf.name_scope('train'):
    train_op = tf.train.GradientDescentOptimizer(0.5).minimize(mean_cross_entropy) # 0.5 is learning rate

sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)
merge_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess.run(init)

for step in range(1000):
    batch_xs, batch_ys  = mnist.train.next_batch(100)
    # 训练输出结果
    _,result = sess.run([train_op, merge_op],feed_dict={xs: batch_xs, ys: batch_ys})
    # merge operation结果添加到图
    writer.add_summary(result, step)
    if step % 50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))


'''
(d:\ProgramData\Anaconda3) E:\pywokspace\market>tensorboard --logdir logs
Starting TensorBoard b'47' at http://0.0.0.0:6006
(Press CTRL+C to quit)

browser http://localhost:6006/
'''
