#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-24 10:45:51
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1+v2

with tf.Session() as sess:
  tf.global_variables_initializer().run()
  # 导出当前计算图的 GraphDef 部分，
  # 只需要这一部分就可以完成从输入层到输出层的计算过程
  graph_def = tf.get_default_graph().as_graph_def()
  
  # 将图中的变量及取值化为常数，同时将图中不需要的节点去掉。
  # 如果只关心程序中定义的某些计算，和这些计算无关的节点就没有必要导出并保存了。
  # 在下面一行代码中，最后一个参数['add']给出了需要保存的节点名称。add节点是上面定义的两个变量相加的
  # 操作。注意这里给出的是计算节点的名称，所以没有后面的:0
  output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
  # 将导出的模型存入文件。
  with tf.gfile.GFile("./models/combined_model.pb", "wb") as f:
    f.write(output_graph_def.SerializeToString())