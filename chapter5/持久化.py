#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-23 22:28:05
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

# 声明两个变量，并计算他们的和。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = v1+v2

# 声明 tf.train.Saver 类用于保存模型。
saver = tf.train.Saver()

with tf.Session() as sess:
  tf.global_variables_initializer().run()
  # 将模型保存到/model/model.ckpt
  saver.save(sess, "./models/model.ckpt")