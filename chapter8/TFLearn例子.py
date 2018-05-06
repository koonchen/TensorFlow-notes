#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-05-06 14:05:21
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
import hues

# 导入 TFLearn
learn = tf.contrib.learn

# 定义 softmax 回归模型
def my_model(features, target):
  # 将预测的目标转换为 one-hot 编码的形式，因为共有三个类别，所以向量长度为3。
  # 经过转化后，第一个类别表示为(1, 0, 0)，第二个是 010 ，第三个是 001
  target = tf.one_hot(target, 3, 1, 0)
  
  # 计算预测值及损失函数。
  logits = tf.contrib.layers.fully_connected(features, 3, tf.nn.softmax)
  loss = tf.losses.softmax_cross_entropy(target, logits)
  
  # 创建优化步骤。
  train_op = tf.contrib.layers.optimize_loss(
    loss, # 损失函数
    tf.train.get_global_step(), # 损失训练步数，并在训练时更新
    optimizer='Adam', # 定义优化器
    learning_rate=0.01 # 定义学习率
  )
  # 返回在给定数据上的预测结果、损失值以及优化步骤。
  return tf.argmax(logits, 1), loss, train_op

# 读取数据并将数据转化成 tf 要求的 float32 格式
iris = datasets.load_iris()
# 划分训练集合和测试集合
x_train, x_test, y_train, y_test = model_selection.train_test_split(
  iris.data, iris.target, test_size=0.2, random_state=0)

x_train, x_test = map(np.float32, [x_train, x_test])
# 封装和训练模型，输出准确率
classifier = SKCompat(learn.Estimator(model_fn=my_model, model_dir="Models/model_1"))
# classifier = learn.Estimator(model_fn=my_model)
classifier.fit(x_train, y_train, steps=800)

y_predicted = [i for i in classifier.predict(x_test)]
score = metrics.accuracy_score(y_test, y_predicted)
hues.info('Accuracy: %.2f%%' % (score * 100))
