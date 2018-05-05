#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-05-05 18:11:35
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf
import numpy as np

# 按时间输入的内容
X = [1, 2]
state = [0.0, 0.0]
# 分开定义不同输入部分的权重。
# np.array 创建了副本，而 np.asarray 不会创建副本
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

# 定义用于输出的全连接层参数。
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 按照时间顺序执行 rnn 的前向传播过程。
for i in range(len(X)):
  # 计算循环体中的全连接层神经网络。
  # np.dot 代表矩阵乘法
  before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
  state = np.tanh(before_activation)
  # 根据当前时刻状态计算最终输出。
  final_output = np.dot(state, w_output) + b_output
  # 输出每个时刻的信息
  print("before activation: ", before_activation)
  print("state: ", state)
  print("output: ", final_output)
