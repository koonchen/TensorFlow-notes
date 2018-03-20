# coding:utf8
import tensorflow as tf

# 下面的例子介绍了入如何通过变量实现神经网络的参数并实现前向传播的过程。
# 声明 w1、w2 两个变量。这里还通过 seed 参数设定了随机种子，这样可以保证每次运行得到的结果是一样的。
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# 暂时将输入的特征向量定义为一个常量。这里 x 是一个 1*2 的矩阵。
x = tf.constant([[0.7,0.9]])

# 通过 3.4.2 小节描述的前向传播算法获得神经网络的输出。
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
# 与 3.4.2 中的计算不同，这里不能直接通过 sess.run(y) 来获得 y 的取值，
# 因为 w1 和 w2 都还没有运行初始化过程。下面的两行分别初始化了 w1 和 w2 两个变量。
sess.run(w1.initializer)
sess.run(w2.initializer)

print(sess.run(y))
sess.close()
