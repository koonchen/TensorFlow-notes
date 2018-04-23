#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-04-23 15:23:55
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

import tensorflow as tf

# 除了使用 tf.Variable 方法创建一个变量，还能使用 tf.get_variable 函数来创建或者获取变量。

# 下面两个定义等价
# name 必须存在
# 首先这个方法会尝试创建一个名为 v 的变量，如果变量存在，将会报错
# 如果想获取某个变量，需要使用 tf.variable_scope 函数建立上下文管理器。
v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))
# name 可有可无
v = tf.Variable(tf.constant(1.0, shape=[1]), name='v')

# 在命名为 foo 的命名空间中创建名为 v 的变量。
with tf.variable_scope("foo"):
  v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))
# 因为 v 已经在 foo 中存在，下面将报错。
# 加上 reuse=True 不再报错。
with tf.variable_scope("foo", reuse=True):
  v = tf.get_variable("v",[1])

# scope 函数器嵌套测试
with tf.variable_scope("root"):
  print(tf.get_variable_scope().reuse)
  with tf.variable_scope("foo", reuse=True):
    print(tf.get_variable_scope().reuse)
    with tf.variable_scope("bar"):
      print(tf.get_variable_scope().reuse)
  print(tf.get_variable_scope().reuse)
# 上面的测试告诉我们，在 tf.variable_scope 中 reuse 参数默认和上层是相同的

# tf.variable_scope 函数生成的上下文管理器也会创建一个 TensorFlow 中的命名空间，在命名空间内创建的变量名称都会带上
# 这个命名空间作为前缀。所以 tf.variable_scope 函数除了控制 tf.get_variable 执行的功能之外，这个函数也提供了一个
# 管理变量命名空间的方式，以下代码显示了如何通过 tf.variable_scope 来管理变量的名称。
v1 = tf.get_variable("v", [1])
print(v1,name)
# 输出 v:0, "v"为变量的名称，":0"表示这个变量是生成变量这个运算的第一个结果。

with tf.variable_scope("foo"):
  v2 = tf.get_variable("v", [1])
  print(v2.name)
  # 输出 foo/v:0 在 tf.variable_scope 中创建的变量，名称前面会加入命名空间的名称，并通过 / 来分割命名空间
  # 的名称和变量的名称。

with tf.variable_scope("foo"):
  with tf.variable_scope("bar"):
    v3 = tf.get_variable("v", [1])
    print(v3.name)
    # 输出 foo/bar/v:0 命名空间可以嵌套，同时变量的名称也会加入所有命名空间的名称作为前缀。

  v4 = tf.get_variable("v1", [1])
  print(v4.name)
  # 输出 foo/v1:0 当命名空间退出之后，变量名称也就不会再加入其前缀了。