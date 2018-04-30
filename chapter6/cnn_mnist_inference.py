#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-27 10:50:06
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

# 这不是论文中提及的标准 LeNet-5
# 配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节点个数
FC_SIZE = 512

# 定义卷积神经网络的前向传播过程。这里添加了一个新的参数 train，用于区分训练过程和测试过程。
# 在这个程序中将用到 dropout 方法， dropout 可以进一步提升模型可靠性并防止过拟合， dropout 过程
# 只在训练时使用。
def inference(input_tensor, train, regularizer):
  # 声明第一层卷积层的变量并实现前向传播过程。这个过程与前文一致。
  # 通过使用不同的命名空间来隔离不同层的变量，这可以让每一层的变量命名只需要考虑在当前层的作用，
  # 而不需要担心重名的问题，和标准的 LeNet-5 模型不大一样，这里的定义输入是 28*28*1 ，使用0填充，
  # 所以输出为 28*28*32
  with tf.variable_scope('layer1-conv1'):
    conv1_weights = tf.get_variable(
      "weight",
      [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],  # 过滤器尺寸、当前层深度、过滤器深度
      initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    conv1_biases = tf.get_variable(
      "bias",
      [CONV1_DEEP],
      initializer=tf.constant_initializer(0.0)
    )
    # 使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且用0填充。
    conv1 = tf.nn.conv2d(
      input_tensor,
      conv1_weights,
      strides=[1, 1, 1, 1],
      padding='SAME'
    )
    # 去线性化
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

  # 实现第二层池化层的前向传播过程。这里选用最大池化层，池化层过滤器的边长为2，
  # 使用全0填充且移动步长为2，这一层的输入是上一层的输出，也就是28x28x32的矩阵。输出为14x14x32的矩阵。
  with tf.name_scope('layer2-pool1'):
    pool1 = tf.nn.max_pool(
      relu1,
      ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding='SAME'
    )
  # 声明第三层卷积层的变量并实现前向传播过程。这一层的输入为14x14x32的矩阵。
  # 输出为14x14x64的矩阵。
  with tf.variable_scope('layer3-conv2'):
    conv2_weights = tf.get_variable(
      "weight",
      [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
      initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    conv2_biases = tf.get_variable(
      "bias",
      [CONV2_DEEP],
      initializer=tf.constant_initializer(0.0)
    )
    # 使用边长为5，深度为64的过滤器，过滤器移动的步长为1，且全部用0填充。
    conv2 = tf.nn.conv2d(
      pool1,
      conv2_weights,
      strides=[1, 1, 1, 1],
      padding='SAME'
    )
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

  # 实现第四层池化层的前向传播过程，这一层和第二层的结构一样。这一层的输入为14x14x64，输出为7x7x64
  with tf.name_scope('layer4-pool2'):
    pool2 = tf.nn.max_pool(
      relu2,
      ksize=[1, 2, 2, 1],
      strides=[1, 2, 2, 1],
      padding='SAME'
    )

  # 将第四层池化层的输出转化为第五层全连接层的输入格式。第四层的输出为 7x7x64 的矩阵，
  # 然后第五层全连接层需要的输入格式为向量，所以在这里需要将这个 7x7x64 的矩阵拉直。
  # pool2.get_shape 函数可以得到第四层输出矩阵的维度而不需要手工计算。注意因为每一层的输入输出都为了
  # 一个 batch 的矩阵，所以这里得到的维度也包含一个 batch 中数据的个数。
  pool_shape = pool2.get_shape().as_list()
  # 计算将矩阵拉直成向量后的长度，这个长度就是矩阵长宽及深度的乘积。
  # 注意这里的 pool_shape[0] 为一个 batch 中数据的个数。
  nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
  # 通过 tf.reshape 函数将第四层的输出变成一个 batch 的向量。
  reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

  # 声明第五层全连接层的变量并实现前向传播过程。这一层的输入时拉直之后的一组向量，
  # 向量长度为 3316，输出是一组长度为 512 的向量。
  # dropout 在训练时会随机将部分节点的输出改为0。dropout 可以避免过拟合问题，从而使得模型在测试数据上
  # 效果更好。dropout 一般在全连接层而不是卷积层或者池化层使用。
  with tf.variable_scope('layer5-fc1'):
    fc1_weights = tf.get_variable(
      "weight",
      [nodes, FC_SIZE],
      initializer = tf.truncated_normal_initializer(stddev=0.1)
    )
    # 只有全连接层的权重需要加入正则化。
    if regularizer != None:
      tf.add_to_collection('losses', regularizer(fc1_weights))
    fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
    if train: 
      fc1 = tf.nn.dropout(fc1, 0.5)

  # 声明第六层全连接层的变量并实现前向传播过程。这一层的输入为一组长度为512的向量，
  # 输出为一组长度为10的向量。这一层的输出通过 Softmax 之后就得到了最后的分类结果。
  with tf.variable_scope('layer6-fc2'):
    fc2_weights = tf.get_variable(
      "weight",
      [FC_SIZE, NUM_LABELS],
      initializer = tf.truncated_normal_initializer(stddev=0.1)
    )
    if regularizer != None:
      tf.add_to_collection('losses', regularizer(fc2_weights))
    fc2_biases = tf.get_variable(
      "bias",
      [NUM_LABELS],
      initializer = tf.constant_initializer(0.1)
    )
    logit = tf.matmul(fc1, fc2_weights) + fc2_biases
  return logit
