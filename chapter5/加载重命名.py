#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-24 09:49:19
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

# 这里声明的变量名称和已经保存的模型中变量的名称不同。
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")

# 如果直接使用 tf.train.Saver() 来加载模型会报变量找不到的错误。
# 使用一个字典来重命名变量可以加载原来的模型，这个字典指定了原来名为 v1 的变量现在名字，同理，v2
saver = tf.train.Saver({"v1":v1, "v2":v2})