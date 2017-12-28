# -*- coding: utf-8 -*-
# -*- author: knock -*-

import tensorflow as tf
import numpy as np
'''
Tensorboard： 可视化逻辑回归训练过程
'''
# 1. numpy造数据
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)： training data 100行1列
noise = np.random.normal(0, 0.1, size=x.shape)      # 噪点
y = np.power(x, 2) + noise                          # shape (100, 1) + noise  ：预测值Y
# 2. 定义tensor 变量
with tf.variable_scope('Inputs'):
    input_x = tf.placeholder(tf.float32, x.shape, name='input_x')
    input_y = tf.placeholder(tf.float32, y.shape, name='input_y')
# 3. 定义神经网络结构input + 1hidden(with RELU激活函数) + ouput
with tf.variable_scope('Net'):
    hidden_1 = tf.layers.dense(input_x, 10, tf.nn.relu, name='hidden_layer')
    output = tf.layers.dense(hidden_1, 1, name='output_layer')

    # 添加到Tensorboard直方图
    tf.summary.histogram('hidden_out', hidden_1)
    tf.summary.histogram('prediction', output)
# 4. 应用mean square 计算cost损失值
loss = tf.losses.mean_squared_error(input_y, output, scope='loss')
# 5. 应用梯度下降优化器优化loss
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
tf.summary.scalar('loss', loss)     # 将loss添加到Tensorboard标量

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('./logs', sess.graph)     # 写Buffer到文件
merge_op = tf.summary.merge_all()

#6. 训练100步
for step in range(100):
    # 训练输出结果
    _, result = sess.run([train_op, merge_op], {input_x: x, input_y: y})
    # merge operation结果添加到图
    writer.add_summary(result, step)

'''
(d:\ProgramData\Anaconda3) E:\pywokspace\market>tensorboard --logdir logs
Starting TensorBoard b'47' at http://0.0.0.0:6006
(Press CTRL+C to quit)

browser http://localhost:6006/
'''

