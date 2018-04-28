#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-26 14:35:52
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

# tf.nn.max_pool 实现了最大池化层的前向传播过程，它的参数和 tf.nn.conv2d 函数类似。
# ksize 提供了过滤器的尺寸， strides 提供了步长， ksize = [1, 3, 3, 1]
# 注意， input 四维含义：batch、height、width、channels
pool = tf.nn.max_pool(
  actived_conv, # 当期节点矩阵
  ksize=[1, 3, 3, 1], # 过滤器尺寸,参考 input
  strides = [1, 2, 2, 1], # 各维步长,参考 input
  padding = 'SAME' 
)