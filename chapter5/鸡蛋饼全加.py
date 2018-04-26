#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-20 09:55:20
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

# 在神经网络结构上，使用激活函数线性化，
# 一个或多个隐藏层使神经网络结构更深；
# 在训练时，使用指数衰减的学习率设置，
# 使用正则化避免过拟合，
# 使用滑动平均使结果更健壮。

# 一套完整的鸡蛋饼，就完成了。
'''
  可扩展性不好，代码大量冗余。
  没有持久化模型，无法再次使用。
  所以这套鸡蛋饼只卖3块...
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集相关的常数
# 输入层的节点数。对于 MNIST 数据集，这个就等于图片的像素。
INPUT_NODE = 784
# 输出层的节点数，这个等于类别的数目。因为在 MNIST 数据集中，需要区分的是 0~9 这几个数字，所以这里是10。
OUTPUT_NODE = 10

# 配置神经网络的参数，
# 隐藏层的节点数，这里使用只有一个隐藏层的网络结构作为样例。这个隐藏层有500个节点。
LAYER1_NODE = 500
# 一个训练 batch 中的训练数据个数，数字越小，训练过程越接近随机梯度下降；数字越大，越接近梯度下降。
BATCH_SIZE = 100
# 基础学习率
LEARNING_RATE_BASE = 0.8
# 学习率的衰减率
LEARNING_RATE_DECAY = 0.99
# 描述模型复杂度的正则化项在损失函数中的系数
REGULARIZATION_RATE = 0.0001
# 训练的轮数
TRAINING_STEPS = 5000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
  # 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果，
  # 在这里定义一个使用 ReLU 激活函数的三层全连接神经网络，通过加入隐藏层实现了多层网络结构，
  # 通过ReLU激活函数实现去线性化，在这个函数中支持传入用于计算参数平均值得类。
  # 这样方便在测试的时候使用滑动平均模型。
  # 当没有提供滑动平均类的时候，直接使用参数当前的取值。
  if avg_class is None:
    # 计算隐藏层的前向传播结果，这里使用 ReLU 激活函数去线性化。
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1)+biases1)
    # 计算输出层的前向传播结果，因为在计算损失函数的时候会一并计算 softmax 函数，
    # 所以这里不需要加入激活函数。并且不加入 softmax 不会影响结果，因为预测时使用的是
    # 不同类别对应节点输出值的相对大小，有没有 softmax 层对最后的分类结果的计算没有
    # 影响。于是在整个神经网络的前向传播中可以不加入最后的 softmax 层。
    return tf.matmul(layer1, weights2)+biases2
  else:
    # 首先使用 avg_class.average 函数计算得出变量的滑动平均值，
    # 然后再计算相应的神经网络前向传播结果。
    layer1 = tf.nn.relu(tf.matmul(input_tensor,
                                  avg_class.average(weights1))+avg_class.average(biases1))
    return tf.matmul(layer1, avg_class.average(weights2))+avg_class.average(biases2)


def train(mnist):
  # 训练模型的过程。
  x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
  y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
  # 生成隐藏层的参数。
  # 正态分布
  weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
  biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
  # 生成输出层的参数
  weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
  biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
  # 计算在当前参数下神经网络前向传播的结果，这里给出的用于计算滑动平均的类是 None，
  # 所以函数不会使用参数的滑动平均值。
  y = inference(x, None, weights1, biases1, weights2, biases2)
  # 定义存储训练轮数的变量，这个变量不需要计算滑动平均值，所以这里这个变量是不可训练的变量 (trainable=False)。
  # 在使用 TensorFlow 训练神经网络时，一般会将训练轮数的变量定位不可训练的参数。
  global_step = tf.Variable(0, trainable=False)
  # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。
  # 这个变量将加快训练早期变量的更新速度。
  variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
  # 在所有的神经网络参数上使用滑动平均，其他辅助变量（比如 global_step ）就不需要，
  # tf.trainable_variables 返回的就是图上集合 GraphKeys.TRAINABLE_VARIABLES 中的元素，这个集合的元素就是
  # 所有没有指定 trainable= False 的参数。
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  # 计算使用滑动平均后的前向传播结果，滑动平均不会改变变量本身，而是维护一个影子变量来记录滑动平均值，所以需要使用
  # 这个滑动平均值时，需要明确调用 average 函数。
  average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
  # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数，这里使用 TensorFlow 中提供的
  # spare_softmax_cross_entropy_with_logits 函数来计算交叉熵。当分类只有一个正确答案时，可以使用这个函数来
  # 加速交叉熵的计算， MNIST 问题的图片中只包含1-9，所以可以使用该方法。这个函数的第一个参数是不加入 softmax 的预测，
  # 第二个参数是正确结果，这里答案时长度为10的数组，而函数需要的是一个数字，所以使用 tf.argmax 函数得出答案编号。
  # 第一次报错，加上参数名。
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
  # 计算在当前 batch 中所有样例的交叉熵平均值。
  cross_entropy_mean = tf.reduce_mean(cross_entropy)

  # 正则化开始，使用 L2正则化损失函数。
  regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
  # 计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失，而不使用偏置项。
  regularization = regularizer(weights1)+regularizer(weights2)
  # 根据正则化公式，接下来计算的是
  # loss = J(θ) + λR(w)
  loss = cross_entropy_mean+regularization
  # 设置指数衰减的学习率
  learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,  # 基础的学习率，随着迭代，更新变量时，在这个基础上递减。
    global_step,  # 当前的迭代轮数
    mnist.train.num_examples/BATCH_SIZE,  # 总共需要的迭代次数
    LEARNING_RATE_DECAY,  # 学习率衰减速度
    staircase=True  # 阶梯状，取整
  )
  # 使用 tf.train.GradientDescentOptimizer 优化算法来优化损失函数。
  # 注意这里包含了交叉熵函数和 L2 正则化损失。
  # 反向传播
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    loss,
    global_step=global_step
  )

  # 在训练神经网络模型时，每过一遍数据，既需要通过反向传播来更新参数，还要更新每一个参数的滑动平均
  # 为了方便完成 TensorFlow 提供了 tf.control_dependencies 和 tf.group 两种机制。
  # 下面的代码和 train_op = tf.group(train_step, variables_averages_op) 等价。
  '''
  with tf.control_dependencies([train_step, variables_averages_op]):
    # 这个 train_op 就是缩写了
    # 1、train_step 反向传播更新参数
    # 2、variable_averages_op 滑动平均全部参数
    train_op = tf.no_op(name='train')
  '''
  train_op = tf.group(train_step, variables_averages_op)
  # 检验使用了滑动平均模型的神经网络前向传播结果是否正确。
  # tf.argmax(average_y, 1)
  # 计算每一个样例的预测结果。
  # 前者代表预测结果，后者代表答案。
  correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
  # 这个运算首先将一个布尔型的数值转换成实数型，然后计算平均值，这个平均值就是模型在这一组数据上的正确率
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  # 初始化会话并开始训练。
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # 准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果
    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    # 准备测试数据，在真实应用中，这部分数据在训练时是不可见的，这个数据只是作为模型优劣的评价。
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}
    # 迭代训练神经网络
    for i in range(TRAINING_STEPS):
      # 每1000轮输出一次在验证训练集上的测试结果。
      if i % 1000 == 0:
        # 计算滑动平均模型在验证数据集上的结果，因为MNIST数据集比较小，所以一次可以处理所有的数据。
        # 为了计算方便，这里没有将数据划分成更小的 batch 。当神经网络模型比较复杂或者验证数据较大
        # 太大的 batch 将使时间过长甚至发生内存溢出。
        validate_acc = sess.run(accuracy, feed_dict=validate_feed)
        # 输出显示验证
        print("after %d training steps, validation accuracy using average model is %g" %
              (i, validate_acc))
        # add test acc
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("after %d training steps, test accuracy using average model is %g" % (i, test_acc),'\n')

      # 产生下一轮 batch 的数据，并运行训练
      xs, ys = mnist.train.next_batch(BATCH_SIZE)
      # 这里的 train_op 完成了滑动平均和反向传播两件事
      sess.run(train_op, feed_dict={x: xs, y_: ys})
    # 训练结束，在测试数据上检测神经网络模型的最终正确率。
    test_acc = sess.run(accuracy, feed_dict=test_feed)
    # 输出显示测试
    print("after %d training steps, test accuracy using average model is %g" %
          (TRAINING_STEPS, test_acc))


def main():
  # 主程序入口
  mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
  train(mnist)


if __name__ == '__main__':
  main()
