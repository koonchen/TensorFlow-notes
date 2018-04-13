#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-13
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

# 当学习率设置为0.001，收敛太慢
import tensorflow as tf
STEPS = 1000
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(y)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(STEPS):
    sess.run(train_op)
    if i % 100 == 99:
      x_value = sess.run(x)
      print("After %s iteration(s): x%s is %f."% (i+1, i+1, x_value))