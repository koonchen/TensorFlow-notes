#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-15 14:11:09
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

# 定义一个变量用于计算滑动平均，这个变量的初始值为0，注意这里手动指定了变量的类型
# 类型为 float32 ，因为所有需要计算滑动平均的变量必须是实数型。
v1 = tf.Variable(0, dtype=tf.float32)
# 这里 step 变量模拟神经网络中迭代的轮数，可以用于动态控制衰减率。
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类 (class)。初始化时给定了衰减率 (0.99) 和控制率的变量 step 。
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作，这里需要给定一个列表，每次执行这个操作时这个列表的变量都会被更新。
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
  # 初始化所有变量。
  tf.global_variables_initializer().run()

  # 通过 ema.average(v1) 获取滑动平均之后变量的取值，在初始化之后变量 v1 的值和 v1 的滑动平均都是0。
  print(sess.run([v1, ema.average(v1)]))

  # 更新变量 v1 的值为 5
  sess.run(tf.assign(v1, 5))

  # 更新 v1 的滑动平均值。衰减率为 min(0.99, (1+step)/(10+step)=0.1)=0.1
  # 所以 v1 的滑动平均会被更新为 0.1*0+0.9*5=4.5
  sess.run(maintain_averages_op)
  print(sess.run([v1, ema.average(v1)]))

  # 更新 step 的值为 10000.
  sess.run(tf.assign(step, 10000))
  # 更新 v1 的值为 10.
  sess.run(tf.assign(v1, 10))
  # 更新 v1 的滑动平均值，衰减率为 min(0.99, (1+step)/(10+step) ≈ 0.999)=0.99
  sess.run(maintain_averages_op)
  print(sess.run([v1, ema.average(v1)]))

  # 再次更新滑动平均值，得到的新滑动值为 0.99*4.555+0.01*10=4.60945
  sess.run(maintain_averages_op)
  print(sess.run([v1, ema.average(v1)]))