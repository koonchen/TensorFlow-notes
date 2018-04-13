#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-13
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

STEPS = 100
global_step = tf.Variable(0)

# 通过 exponential_decay 函数生成学习率。
learning_rate = tf.train.exponential_decay(0.1,global_step,1,0.96,staircase=True)

x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)

# 使用指数衰减的学习率，在 minimize 函数中传入 global_step 将自动更新
# global_step 参数，从而使得学习率也得到相应更新。
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(y,global_step=global_step)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(STEPS):
    sess.run(learning_step)
    if i % 10 == 9:
      LEARNING_RATE_value = sess.run(learning_rate)
      x_value = sess.run(x)
      print("After %s iteration(s): x%s is %f, learning rate is %f."% 
        (i+1, i+1, x_value, LEARNING_RATE_value))