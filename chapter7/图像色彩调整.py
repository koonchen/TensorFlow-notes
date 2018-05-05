#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-05-03 15:20:49
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("../datasets/cat.jpg",'rb').read()

with tf.Session() as sess:
  # 对图像进行解码
  img_data = tf.image.decode_jpeg(image_raw_data)
  # 进行图片处理
  # 亮度。
  adjusted = tf.image.adjust_brightness(img_data, -0.5)
  # 对比度。
  adjusted = tf.image.adjust_contrast(img_data, -5)
  # 相色
  adjusted = tf.image.adjust_hue(img_data, 0.3)
  # 饱和度
  adjusted = tf.image.adjust_saturation(img_data, -5)
  # 将图像的三维矩阵中的数字均变成0，方差变为1
  adjusted = tf.image.per_image_standardization(img_data)
  # 将图像缩小，这样可以让可视化的标注框更加清楚。
  # 方法取 1 将黑屏。
  img_data = tf.image.resize_images(img_data, [180, 267], method=0)
  # tf.image.draw_bounding_boxes 函数要求图像矩阵中的数字为实数，所以需要先将图像矩阵转化为实数类型。
  # tf.image.draw_bounding_boxes 函数图像的输入为一个 batch 数据，也就是将多张图像组成的四维矩阵，
  # 所以需要将解码之后的图像矩阵加一维。
  batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
  # 给出每一张图像的所有标注框。一个标注框有四个数字，分别代表 ymin, xmin, ymax, xmax
  # 注意这里给出的数组都是图像的相对位置。
  # [0.35, 0.47, 0.5, 0.56] 代表从 (63, 125) 到 (90, 150) 的图像
  boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
  # 在图像上加入标注框
  result = tf.image.draw_bounding_boxes(batched, boxes)
  # 将结果重新编码显示
  # 输出选择 result[0]
  cat = np.asarray(result[0].eval(), dtype='uint8')
  plt.imshow(cat)
  plt.show()
