#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-25 09:04:27
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

# tf.train.NewCheckpointReader 可以读取 checkpoint 文件中保存的所有变量。
reader = tf.train.NewCheckpointReader("./models/model.ckpt")

# 获取所有变量列表。这个是一个从变量名到变量维度的字典。
all_variables = reader.get_variable_to_shape_map()
for variable_name in all_variables:
  # variable_name 为变量名称， all_variables[variable_name] 为变量的维度
  print(variable_name, all_variables[variable_name])

# 获取名称为 v1 的变量取值
print("value for variable v1 is ", reader.get_tensor("v1"))