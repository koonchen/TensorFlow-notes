#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-26 09:27:03
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

# 通过 tf.get_variable 的方式创建过滤器的权重和偏置项变量。
# 上面介绍了卷积层的参数个数只和过滤器的尺寸、深度、当前层的节点矩阵深度有关，
# 这里的参数变量是一个四维矩阵，前面两个维度代表了过滤器的尺寸，第三个维度表示当前层的深度，第四过滤器深度。
# 5  过滤器长
# 5  过滤器宽
# 3  当前层深度
# 16 过滤器深度
filter_weight = tf.get_variable(
  'weights', 
  [5, 5, 3, 16], 
  initializer = tf.truncated_normal_initializer(stddev=0.1)
)

# 和卷积层的权重类似，当前层矩阵上不同位置的偏置项也是共享的，所以总共有下一层深度个不同的偏置项。
# 这里的 16 是过滤器的深度，也是下一层节点矩阵的深度。
biases = tf.get_variable(
  'biases', 
  [16], 
  initializer = tf.constant_initializer(0.1)
)

# tf.nn.conv2d 提供了一个非常方便的函数来实现卷积层前向传播的算法。这个函数的第一个输入为当前层的节点矩阵。
# 注意这个矩阵是一个四维的矩阵，后面三个维度对应一个节点矩阵，第一维对应一个输入 batch 。比如在输入侧，
# input[0, :, :, :] 表示第一章图片， input[1, :, :, :] 表示第二张图片，以此类推。
# tf.nn.conv2d 的第二个参数提供了卷积层的权重，第三个参数为不同维度上的步长。
# 虽然第三个参数提供的是一个长度为 4 的数组，但是第一维和最后一维的数字要求一定是1。这是因为卷积层的步长
# 只对常和宽有效。最后一个参数是填充的方法， TensorFlow 中提供 SAME 或是 VALID 两种选择。其中 SAME 表示
# 添加全0填充，VALID 表示不添加。
conv = tf.nn.conv2d(
  input, # 输入图片， [0, :, :, :] 代表第一张图片
  filter_weight, # 代表权重
  strides=[1, 1, 1, 1], # 在不同维度上的步长
  padding = 'SAME' # 填充 0
)

# tf.nn.bias_add 提供了一个方便的函数给每一个节点加上偏置项，注意这里不能直接使用加法，因为矩阵上不同
# 位置的节点都需要加上同样的偏置项。虽然下一层神经网络大小是2x2，但是偏置项只有一个数(深度为1)，而2x2矩阵中
# 每一个值都需要加入这个偏置项。
bias = tf.nn.bias_add(conv, biases)
# 将计算结果通过 ReLU 激活函数完成去线性化。
actived_conv = tf.nn.relu(bias)