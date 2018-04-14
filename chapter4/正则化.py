#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-14 09:53:02
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 生成模拟数据集
# 模拟需要分割的数据
dataset_size = 200
data = []
label = []
# seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，
# 如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
np.random.seed(0)
# 以原点为圆心，半径为1的圆把散点划分成红蓝两部分，并加入随机噪音。
for i in range(dataset_size):
  # x1 从-1到1 随机取样 左闭右开
  x1 = np.random.uniform(-1, 1)
  # x2 从0到2 随机取样
  x2 = np.random.uniform(0, 2)
  if x1**2+x2**2 <= 1:
    data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
    label.append(0)
  else:
    data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
    label.append(1)
data = np.hstack(data).reshape(-1, 2)
label = np.hstack(label).reshape(-1, 1)
plt.scatter(data[:, 0], data[:, 1], c=np.squeeze(label), cmap="RdBu", vmin=-.2, vmax=1.2, edgecolors="white")
# plt.show()

# 定义带L2正则化的损失函数的计算方法
# 获取一层神经网络边上的权重，并将这个权重的 L2 正则化损失加入名称为 'loss' 的集合中。
def get_weight(shape, var_lambda):
  # 生成一个变量
  var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
  # add_to_collection 函数将这个新生成变量的 L2 正则化损失项加入集合。
  # 这个函数的第一个参数 ‘loss’ 是集合的名字，第二个参数是要加入这个集合的内容。
  tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(var_lambda)(var))
  # 返回生成的变量
  return var

x = tf.placeholder(tf.float32,shape=(None, 2))
y_ = tf.placeholder(tf.float32,shape=(None, 1))
# 定义了每一层网络中节点的个数
layer_dimension = [2, 10, 5, 3, 1]
# 神经网络的层数
n_layers = len(layer_dimension)
# 这个变量维护前向传播时最深层的节点，开始的时候就是输入层。
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]
# 通过一个循环来生成5层全连接的神经网络结构。
for i in range(1, n_layers):
  # layer_dimension[i] 为下层的节点个数。
  out_dimension = layer_dimension[i]
  # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图的集合。
  weight = get_weight([in_dimension, out_dimension], 0.003)
  bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
  # 使用 ReLU 激活函数
  # 视同 relu 将是折线， elu 将是平滑的线
  cur_layer = tf.nn.elu(tf.matmul(cur_layer, weight)+bias)
  # 进入下一层之前将下一层的节点个数更新为当前层的节点个数
  in_dimension = layer_dimension[i]

# 在定义神经网络前向传播的同时已经将所有的 L2 正则化损失加入到图上的集合
# 这里只需要计算刻画模型在训练数据上表现的损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
# 将均方差损失函数加入损失集合
tf.add_to_collection('losses', mse_loss)

# get_collection 返回一个列表，这个列表是所有集合的元素，在这个样例中，
# 这些元素就是损失函数的不同部分，将他们加起来就可以得到最终想要的损失函数。
loss = tf.add_n(tf.get_collection('losses'))
'''
# 训练以及结果显示
# 定义训练的目标函数mse_loss，训练次数及训练模型
# 不带正则项的结果
train_op = tf.train.AdamOptimizer(0.001).minimize(mse_loss)
TRAINING_STEPS = 10000
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  for i in range(TRAINING_STEPS):
    sess.run(train_op, feed_dict={x: data, y_: label})
    if i % 1000 == 1000 - 1:
      print("After %d steps, mse_loss: %f" % (i,sess.run(mse_loss, feed_dict={x: data, y_: label})))
  # 画出训练后的分割曲线
  xx, yy = np.mgrid[-1.2:1.2:.01, -0.2:2.2:.01]
  grid = np.c_[xx.ravel(), yy.ravel()]
  probs = sess.run(cur_layer, feed_dict={x:grid})
  probs = probs.reshape(xx.shape)

plt.scatter(data[:,0], data[:,1], c=np.squeeze(label), cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)
plt.show()
'''
# 带正则项的结果
# 定义训练的目标函数loss，训练次数及训练模型
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
TRAINING_STEPS = 10000

with tf.Session() as sess:
  tf.global_variables_initializer().run()
  for i in range(TRAINING_STEPS):
    sess.run(train_op, feed_dict={x: data, y_: label})
    if i % 1000 == 1000 - 1:
      print("After %d steps, loss: %f" % (i, sess.run(loss, feed_dict={x: data, y_: label})))

  # 画出训练后的分割曲线       
  xx, yy = np.mgrid[-1:1:.01, 0:2:.01]
  grid = np.c_[xx.ravel(), yy.ravel()]
  probs = sess.run(cur_layer, feed_dict={x:grid})
  probs = probs.reshape(xx.shape)

plt.scatter(data[:,0], data[:,1], c=np.squeeze(label), cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)
plt.show()
