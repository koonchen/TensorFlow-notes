#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-05-04 14:18:47
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

# 声明一个先进先出的队列，队列中最多100个元素，类型为实数。
# 定义将操作的队列
queue = tf.FIFOQueue(100, "float")
# 定义队列的入队操作
# 定义入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 使用 tf.train.QueueRunner 来创建多个线程运行队列的入队操作。
# QueueRunner 的第一个参数给出了被操作的队列， [enqueue_op]*5
# 表示需要启动 5 个线程，每个线程中运行的是 enqueue_op 操作。
qr = tf.train.QueueRunner(queue, [enqueue_op]*5)

# 将定义过 QueueRunner 加入 TensorFlow 计算图上指定的集合。
# add_queue_runner 函数没有指定集合，则加入默认集合 QUEUE_RUNNERS 。
# 下面的函数就是将刚刚定义的 qr 加入默认的 QUEUE_RUNNERS 集合。
tf.train.add_queue_runner(qr)
# 定义出队操作。
out_tensor = queue.dequeue()

with tf.Session() as sess:
  # 使用 tf.train.Coordinator 来协同启动的线程。
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  # 获取队列中的取值
  for _ in range(3):
    print(sess.run(out_tensor)[0])
  # 使用 tf.train.Coordinator 来停止所有的线程 
  coord.request_stop()
  # 等待所有线程退出。
  coord.join(threads)
