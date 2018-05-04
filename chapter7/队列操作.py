#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-05-03 16:56:23
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

# 创建一个先进先出队列，指定队列中最多可以保存两个元素，并指定类型为整数。
q = tf.FIFOQueue(2, "int32")
# 使用 enqueue_many 函数来初始化队列中的元素。和元素初始化类似，在使用队列之前需要明确的调用这个
# 初始化过程。
init = q.enqueue_many(([0, 10],))
# 使用 Dequeue 函数将队列中的第一个元素出队列。这个元素的值将被存在变量 x 中。
x = q.dequeue()
# 将得到的值加 1
y = x+1
# 将加 1 的值重新加入队列
q_inc = q.enqueue([y])

with tf.Session() as sess:
  # 运行初始化队列的操作。
  init.run()
  for _ in range(5):
    # 运行 q_inc 将执行数据出队列、出队的元素加 1、重新加入队列的整个过程。
    v, _ = sess.run([x, q_inc])
    # 打印出队元素的取值。
    print(v)
