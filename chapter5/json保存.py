#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-24 22:47:38
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

# 定义变量相加的计算、
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result1 = v1+v2

saver = tf.train.Saver()
# 通过 export_meta_graph 函数导入 TensorFlow 计算图的元图，并保存为 json 格式。
saver.export_meta_graph("./models/jsonModel.ckpt.meda.json", as_text=True)
