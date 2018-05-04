#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-05-03 14:04:24
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

# 在之前的几章中都采用了直接使用图像原始的像素矩阵，这一章将对图像进行预处理。
# 图像编码处理。
# matplotlib.pyplot 是一个 python 的画图工具。在这一节中将使用这个工具来可视化 TensorFlow 处理后的图像。

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("../datasets/cat.jpg",'rb').read()

with tf.Session() as sess:
  # 将图像使用 jpeg 格式解码从而得到对应的三维矩阵。 TensorFlow 还提供了 tf.image.decode_png 函数对
  # png 格式的图像进行解码，解码之后的结果为一个张量，在使用它之前还需要明确调用运行的过程。
  img_data = tf.image.decode_jpeg(image_raw_data)
  # plt.imshow(img_data.eval())
  # plt.show()
  # 将数据的类型转化成实数方便下面的案例程序对图像进行处理。

  # 这句不注释下面的图片将变为全黑
  # img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

  # 重新定义大小
  resized = tf.image.resize_images(img_data, [300, 300], method=0)
  # 裁剪大小
  # 通过 tf.image.central_crop 函数可以按比例裁剪。这个函数的第一个参数为原始图像，第二个参数为调整比例。
  central_crop = tf.image.central_crop(img_data, 0.5)

  # 图像翻转
  # 将图像上下翻转
  flipped_up_down = tf.image.flip_up_down(img_data)
  # 将图像左右翻转
  filpped_left_right = tf.image.flip_left_right(img_data)
  # 将图像沿对角线翻转
  transposed = tf.image.transpose_image(img_data)
  # 随机进行上下/左右翻转
  # flipped = tf.image.random_flip_up_down(img_data)
  flipped = tf.image.random_flip_left_right(img_data)

  cat = np.asarray(flipped.eval(), dtype='uint8')
  plt.imshow(cat)
  plt.show()
  # print(img_data.get_shape())
  

  # # 将表示一张图像的三维矩阵重新按照 jpeg 格式编码并放入文件中，打开这张图像，可以得到和原始图像一样的
  # # 图像。
  # encode_image = tf.image.encode_jpeg(img_data)
  # with tf.gfile.GFile("../datasets/cat.jpg", "wb") as f:
  #   f.write(encode_image.eval())
