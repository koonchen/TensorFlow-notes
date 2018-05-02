#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-29 13:34:13
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

# 直接使用 TensorFlow 原始 API 实现卷积层
with tf.variable_scope(scope_name):
  weights = tf.get_variable("weights", ...)
  biases = tf.get_variable("bias", ...)
  conv = tf.nn.conv2d(...)
  relu = tf.nn.relu(tf.nn.bias_add(conv, biases))

# 使用 TensorFlow-Slim 实现卷积层。通过 TensorFlow-Slim 可以在一行中实现一个卷积层的前向传播。
# slim.conv2d 函数的三个参数是必填的，第一个参数为输入节点矩阵，第二个参数是卷积层过滤器的深度，
# 第三个参数是过滤器的尺寸。可选的参数有过滤器的步长、是否使用全0填充、激活函数的选择以及变量的命名空间
net = slim.conv2d(input, 32, [3, 3])

# 下面实现一个 Inception 模块
# slim.arg_scope 函数可以用于设置默认的参数取值。 slim.arg_scope 函数的第一个参数是一个函数列表，
# 这个函数列表中的函数将使用默认的参数取值，比如通过下面的定义，调用 slim.conv2d(net, 320, [1, 1])
# 函数会自动加上 stride = 1 和 padding = 'SAME' 的参数。如果在函数调用时指定了 stride ，那么这里
# 设置的默认值就不会再使用。通过这种方式可以进一步减少冗余代码。
with slim.arg_scope([sliml.con2d, slim.max_pool2d, slim.avg_pool2d], stride=1 ,padding='SAME'):
  # 此处省略了 Inception-v3 模型中其他的网络结构而直接实现最后面框中的 Inception 结构。
  # 假设输入图片经过的神经网络前向传播的结果而保存在变量 net 中。
  net = 上一次的输入节点矩阵
  # 为一个 Inception 模块声明一个统一的变量命名空间
  with tf.variable_scope('Mixed_7c'):
    # 给 Inception 模块中每一条路径声明一个命名空间
    with tf.variable_scope('Branch_0'):
      # 实现一个过滤器边长为 1，深度为 320 的卷积层。
      branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
    # Inception 模块中第二条路径。这条计算路径上的结构本身也是一个 Inception 结构。
    with tf.variable_scope('Branch_1'):
      branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
      # tf.concat 函数可以将多个矩阵拼接在一起。tf.concat 函数的第一个参数指定了拼接的维度，
      # 这里给出的"3"代表了矩阵在深度这个维度上进行拼接。
      branch_1 = tf.concat(3, [
        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0a_1x3'),
        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0a_3x1')
      ])
    # Inception 模块中第三条路径。此计算路径也是一个 Inception 结构。
    with tf.variable_scope('Branch_2'):
      branch_2 = slim.conv2d(
        net, 448, [1, 1], scope='Conv2d_0a_1x1'
      )
      branch_2 = slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0a_1x3')
      branch_2 = tf.concat(3, [
        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0a_1x3'),
        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0a_3x1')
      ])
    # Inception 模块中第四条路径。
    with tf.variable_scope('Branch_3'):
      branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
      branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0a_1x1')
    # 当前 Inception 模块的最后输出是由上面四个计算结果拼接得到的。
    # 这里的 3 表示在第三维度上进行连接。
    net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
