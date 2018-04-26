#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-25 10:18:24
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

# 神经网络训练
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载 mnist_inference.py 中定义的常量和前向传播的函数
import mnist_inference 
# 配置神经网络的参数。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARATION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./models/"
MODEL_NAME = "model.ckpt"

def train(mnist):
  # 定义输入输出 placeholder
  x = tf.placeholder(
    tf.float32, 
    [None, mnist_inference.INPUT_NODE], 
    name = 'x-input'
  )
  y_ = tf.placeholder(
    tf.float32,
    [None, mnist_inference.OUTPUT_NODE], 
    name = 'y_input'
  )
  regularizer = tf.contrib.layers.l2_regularizer(REGULARATION_RATE)
  # 直接使用 mnist_inference.py 中定义的前向传播过程。
  y = mnist_inference.inference(x, regularizer)
  # 这个值是不需要训练的
  global_step = tf.Variable(0, trainable=False)

  # 滑动平均值
  variable_averages = tf.train.ExponentialMovingAverage(
    MOVING_AVERAGE_DECAY, 
    global_step
  )
  variable_averages_op = variable_averages.apply(tf.trainable_variables())
  # 损失函数
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
  cross_entropy_mean = tf.reduce_mean(cross_entropy)
  loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
  # 指数衰减学习率
  learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE, 
    global_step, 
    mnist.train.num_examples/BATCH_SIZE, 
    LEARNING_RATE_DECAY
  )
  # 训练过程
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, 
    global_step = global_step)
  # 合并更新操作
  train_op = tf.group(train_step, variable_averages_op)
  # 持久化类
  saver = tf.train.Saver()
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成。
    for i in range(TRAINING_STEPS):
      xs, ys = mnist.train.next_batch(BATCH_SIZE)
      _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_:ys})
      # 每 1000 轮保存模型
      if i%1000 == 0:
        # 输出当前的训练情况，这里只输出模型在当前训练 batch 上的损失函数大小。
        # 通过损失函数大小可以大概了解训练情况。在验证数据集上正确率信息会有一个单独的程序来生成。
        print("After %d training steps, loss on training batch is %g." % (step, loss_value))
        # 保存当前模型，注意这里给出了 global_step 参数，这样可以让每一个保存模型的文件名末尾加上轮数。
        # 比如 model.ckpt-1000
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
  mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
  train(mnist)

if __name__ == '__main__':
  tf.app.run()
