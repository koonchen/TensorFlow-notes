#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-05-03 09:30:37
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 生成整数型的属性
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串类型的属性
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets("../datasets/MNIST_data", dtype=tf.uint8, one_hot=True)
images = mnist.train.images

# 训练数据所对应的正确答案，可以作为一个属性保存在 TFRecord 中。
labels = mnist.train.labels
# 训练数据的图像分辨率，这可以作为 Example 中的一个属性。
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出 TFRecord 文件的地址
# 需要提前创建 Records 文件夹
filename = "Records/output.tfrecords"
# 创建一个 writer 来写 TFRecord 文件。
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
  # 将图像矩阵转化成一个字符串
  image_raw = images[index].tostring()
  # 将一个样例转换为 Example Protocol Buffer ，将所有的信息写入这个数据结构。
  example = tf.train.Example(features=tf.train.Features(feature={
    'pixels': _int64_feature(pixels),
    'labels': _int64_feature(np.argmax(labels[index])),
    'image_raw': _bytes_feature(image_raw)
  }))
  # 将一个 Example 写入 TFRecord 文件
  writer.write(example.SerializeToString())
writer.close()

# 下面写读取 TFRecord 文件

# 创建一个 reader 来读取 TFRecord 文件中的样例。
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表。
# tf.train.string_input_producer 函数
filename_queue = tf.train.string_input_producer(["Records/output.tfrecords"])

# 从文件中读出一个样例。也可以使用 read_up_to 函数一次性读取多个样例。
_, serialized_example = reader.read(filename_queue)
# 解析读入的一个样例。如果需要解析多个样例，可以用 parse_example 函数
features = tf.parse_single_example(
  serialized_example,
  features={
    # TensorFlow 提供两种不同的属性解析方法，一种是 tf.FixedLenFeature , 这种方法的结果是一个 Tensor，
    # 另一种方法是 tf.VarLenFeature ,这种方法得到的结果是 SpareTensor ,用于处理稀疏数据。这里解析数据
    # 的格式需要和上面程序写入数据的格式一致。
    'image_raw': tf.FixedLenFeature([], tf.string),
    'pixels': tf.FixedLenFeature([], tf.int64),
    'labels': tf.FixedLenFeature([], tf.int64)
  }
)
# tf.decode_raw 可以将字符串解析成图像对应的像素数组。
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['labels'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
# 启动多线程来处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)

# 每次运行可以读取 TFRecord 文件中的一个样例。当所有样例都读取完成以后，在此样例中重头再读。
for i in range(10):
  image, label, pixel = sess.run([images, labels, pixels])
  print(label, pixel)
