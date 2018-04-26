#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-24 09:15:31
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

saver = tf.train.import_meta_graph("./models/model.ckpt.meta")
with tf.Session() as sess:
  saver.restore(sess, "./models/model.ckpt")
  # 通过张量的名称来获取张量
  print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))