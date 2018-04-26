#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-24 10:04:58
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
# 在没有申明滑动平均模型时只有一个变量v,所以下面的语句会输出 "v:0"
for variables in tf.global_variables():
  print(variables.name, "\n")

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
# 在申明滑动平均模型之后，TensorFlow 会自动生成一个影子变量。
# v/ExponentialMoving Average。下面会输出：
# "v:0" 和 "v/ExponentialMovingAverage:0"
for variables in tf.global_variables():
  print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  sess.run(tf.assign(v, 10))
  sess.run(maintain_averages_op)
  # 保存时，TensorFlow 会将 v:0 和 v:ExponentialMovingAverage:0 两个变量都保存下来。
  saver.save(sess, "./models/emaModel.ckpt")
  print(sess.run([v, ema.average(v)]))