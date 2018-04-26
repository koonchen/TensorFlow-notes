#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-24 10:19:44
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf
v = tf.Variable(0, dtype=tf.float32, name="v")
# 通过变量重命名将原来变量 v 的滑动平均值直接赋值给 v
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
  saver.restore(sess, "./models/emaModel.ckpt")
  print(sess.run(v))

# 为了方便加载时重命名滑动平均变量， tf.train.ExponentialMovingAverage 类提供了 variables_to_restore
# 函数来生成 tf.train.Saver 类所需要的变量重命名字典
v = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(0.99)
# 通过使用 variables_to_restore 函数可以直接生成上面代码提供的字典
print(ema.variables_to_restore())
# 输出{'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
  saver.restore(sess, "./models/emaModel.ckpt")
  print(sess.run(v))
  # 输出 v 的滑动平均值