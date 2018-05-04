#!/usr/bin/env python
# coding:utf8
# @Date    : 2018-05-04 15:11:48
# @Author  : Koon
# @Link    : zpkoon.xyz
# When I wrote this, only God and I understood what I was doing. Now, God only knows.

# 生成文件存储的样例数据。
import tensorflow as tf

# 创建 TFRecord 文件的帮助函数。
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 模拟海量数据情况下将数据写入不同的文件， num_shards 定义了总共写入多少个文件，而 instances_per_shard
# 定义了每个文件中有多少个数据。
num_shards = 2
instances_per_shard = 2
for i in range(num_shards):
  # 将数据分为多个文件时，可以将不同文件以类似 0000n-of-0000m 的后缀区分。
  filename = ('data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
  writer = tf.python_io.TFRecordWriter(filename)
  # 将数据封装成 Example 结构并写入 TFRecord 文件。
  for j in range(instances_per_shard):
    # Example 结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本。
    example = tf.train.Example(features=tf.train.Features(feature={
      'i': _int64_feature(i),
      'j': _int64_feature(j)
    }))
    writer.write(example.SerializeToString())
  writer.close()

# 读取文件
files = tf.train.match_filenames_once("data.tfrecords-*")
# 将 shuffle 设置为 false ，避免随机打乱读文件的顺序，在真实问题中将设置成 True
filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
  serialized_example,
  features={
    'i': tf.FixedLenFeature([], tf.int64),
    'j': tf.FixedLenFeature([], tf.int64)
  }
)
with tf.Session() as sess:
  sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
  print(sess.run(files))
  # 声明 Coordinator 类来协同不同的线程。
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  # 多次执行获取数据的操作。
  for i in range(6):
    print(sess.run([features['i'], features['j']]))
  coord.request_stop()
  coord.join(threads)

# 组合训练数据
example, label = features['i'], features['j']
batch_size = 2
capacity = 1000 + 3 * batch_size
example_batch, label_batch = tf.train.batch(
  [example, label], batch_size=batch_size, capacity=capacity
)
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  tf.local_variables_initializer().run()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  for i in range(3):
    cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
    print(cur_example_batch, cur_label_batch)
  coord.request_stop()
  coord.join(threads)
