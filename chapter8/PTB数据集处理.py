#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-05-05 20:01:27
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf
import reader

# 读取数据打印前 100 位数据
DATA_PATH = "../datasets/PTB_data"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
print(len(train_data))
print(train_data[:100])

# 将训练数据组织成 batch 大小为 4、截断长度为 5 的数据组。
result = reader.ptb_producer(train_data, 4, 5)
# 通过队列依次读取batch。
with tf.Session() as sess:
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  for i in range(3):
    x, y = sess.run(result)
    print ("X%d: "%i, x)
    print ("Y%d: "%i, y)
  coord.request_stop()
  coord.join(threads)
