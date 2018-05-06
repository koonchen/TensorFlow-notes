#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-05-06 15:29:14
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import hues
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat

# 加载 matplotlib 工具包，使用该工具可以对预测的 sin 函数曲线进行绘图。
import matplotlib as mpl

mpl.use('Agg')
from matplotlib import pyplot as plt

learn = tf.contrib.learn

HIDDEN_SIZE = 30 # LSTM 中隐藏层的节点个数
NUM_LAYERS = 2 # LSTM 的层数
TIMESTEPS = 10 # 循环神经网络的截断长度
TRAINING_STEPS = 3000 # 训练轮数
BATCH_SIZE = 32 # batch 大小

TRAINING_EXAMPLES = 10000 # 训练数据的个数
TESTING_EXAMPLES = 1000 # 测试数据个数
SAMPLE_GAP = 0.01 # 采样间隔

def generate_data(seq):
  X = []
  y = []
  # 序列的第 i 项和后面的 TIMESTEPS-1 项合在一起作为输入；第 i+TIMESTEPS 项作为输出。即用 sin 函数
  # 前面的 TIMESTEPS 个点的信息，预测第 i + TIMESTEPS 个点的函数值。
  for i in range(len(seq) - TIMESTEPS - 1):
    X.append([seq[i: i+TIMESTEPS]])
    y.append([seq[i+TIMESTEPS]])
  return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def lstm_model(X, y):
  lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
  cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)
  
  output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
  output = tf.reshape(output, [-1, HIDDEN_SIZE])
  
  # 通过无激活函数的全联接层计算线性回归，并将数据压缩成一维数组的结构。
  predictions = tf.contrib.layers.fully_connected(output, 1, None)
  
  # 将predictions和labels调整统一的shape
  labels = tf.reshape(y, [-1])
  predictions=tf.reshape(predictions, [-1])
  
  loss = tf.losses.mean_squared_error(predictions, labels)
  
  train_op = tf.contrib.layers.optimize_loss(
    loss, tf.contrib.framework.get_global_step(),
    optimizer="Adagrad", learning_rate=0.1)

  return predictions, loss, train_op

# 封装之前定义的lstm。
# regressor = SKCompat(learn.Estimator(model_fn=lstm_model,model_dir="Models/model_2"))
regressor = learn.Estimator(model_fn = lstm_model)
# 生成数据。
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(
  0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
  test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))

# 拟合数据。
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

# 计算预测值。
predicted = [[pred] for pred in regressor.predict(test_X)]

# 计算MSE。
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print ("Mean Square Error is: %f" % rmse[0])
