#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-24 11:08:19
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
  model_filename = "./models/combined_model.pb"
  with gfile.FastGFile(model_filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  # 将 graph_def 中保存的图加载到当前图中。return_elements=["add:0"]给出了返回的张量名称。在保存的时候
  # 给出的是计算节点的名称，所以是 "add"。加载的时候给出的是张量的名称，所以是 add:0 。
  result = tf.import_graph_def(graph_def, return_elements=["add:0"])
  print(sess.run(result))
  # 这个加法计算已经定义为 1+2