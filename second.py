# -*- coding:utf-8 -*-

# 导入数据集
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# Variables
batch_size = 100
total_steps = 5000
#
dropout_keep_prob = 0.5
steps_per_test = 100

''' 
使用截尾正态分布函数 truncated_normal()来生成初始化张量
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
从截断的正态分布中输出随机值

shape: 输出的张量的维度尺寸。
mean: 正态分布的均值。
stddev: 正态分布的标准差。
dtype: 输出的类型。
seed: 一个整数，当设置之后，每次生成的随机数都一样。
name: 操作的名字。
'''


def weight(shape, stddev=0.1, mean=0):
    initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)


# constant()方法是用于生成常量的方法
def bias(shape, value=0.1):
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


'''
conv2d()方法进行卷积操作。
input，指需要做卷积的输入图像，它要求是一个 Tensor，具有 [batch_size, in_height, in_width, in_channels] 这样的 shape，
具体含义是 [batch_size 的图片数量, 图片高度, 图片宽度, 输入图像通道数]，注意这是一个 4 维的 Tensor，要求类型为 float32 和 float64 其中之一。
filter，相当于 CNN 中的卷积核，它要求是一个 Tensor，具有 [filter_height, filter_width, in_channels, out_channels] 这样的shape，
具体含义是 [卷积核的高度，卷积核的宽度，输入图像通道数，输出通道数（即卷积核个数）]，要求类型与参数 input 相同，有一个地方需要注意，第三维 in_channels，就是参数 input 的第四维。
strides，卷积时在图像每一维的步长，这是一个一维的向量，长度 4，具有 [stride_batch_size,stride_in_height, stride_in_width, stride_in_channels]这样的shape
第一个元素代表在一个样本的特征图上移动，第二三个元素代表在特征图上的高、宽上移动，第四个元素代表在通道上移动。
padding，string 类型的量，只能是 SAME、VALID 其中之一，这个值决定了不同的卷积方式。
use_cudnn_on_gpu，布尔类型，是否使用 cudnn 加速，默认为true。
返回的结果是 [batch_size, out_height, out_width, out_channels] 维度的结果。
'''


def conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(input, filter, strides=strides, padding=padding)


''' 
池化层往往在卷积层后面，通过池化来降低卷积层输出的特征向量，同时改善结果。
value，需要池化的输入，一般池化层接在卷积层后面，所以输入通常是 feature map，依然是 [batch_size, height, width, channels] 这样的shape。
ksize，池化窗口的大小，取一个四维向量，一般是 [batch_size, height, width, channels]，因为我们不想在 batch 和 channels 上做池化，所以这两个维度设为了1。
strides，和卷积类似，窗口在每一个维度上滑动的步长，一般也是 [stride_batch_size, stride_height, stride_width, stride_channels]。
padding，和卷积类似，可以取 VALID、SAME，返回一个 Tensor，类型不变，shape 仍然是 [batch_size, height, width, channels] 这种形式。
'''


def max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding)


# Initial
# 初始化
x = tf.placeholder(tf.float32, shape=[None, 784])
y_label = tf.placeholder(tf.float32, shape=[None, 10])

# reshape()
# tf.reshape(tensor, shape, name=None)
# 函数的作用是将tensor变换为参数shape的形式。
# 其中shape为一个列表形式，特殊的一点是列表中可以存在-1。-1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个-1。
# （当然如果存在多个-1，就是一个存在多解的方程了）

x_reshape = tf.reshape(x, [-1, 28, 28, 1])

# Layer1
# 第一层首先初始化w和b
# conv2d卷积操作，然后应用reLU激活函数进行非线性转换，然后max pooling操作。

w_conv1 = weight([5, 5, 1, 32])
b_conv1 = bias([32])
h_conv1 = tf.nn.relu(conv2d(x_reshape, w_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1)

# Layer2
# 同上

w_conv2 = weight([5, 5, 32, 64])
b_conv2 = bias([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2)

# Layer3
# 密集连接层
# 现在，图片尺寸减小到7×7，我们再加入一个有 1024 个神经元的全连接层，用于处理整个图片。
# 我们把池化层输出的张量 reshape 成一些向量，乘上权重矩阵，加上偏置，然后对其使用 ReLU。

w_fc1 = weight([7 * 7 * 64, 1024])
b_fc1 = bias([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# 为了减少过拟合，在隐藏层和输出层之间加人dropout操作。
# 用来代表一个神经元的输出在dropout中保存不变的概率。
# 在训练的过程启动dropout，在测试过程中关闭dropout
# Dropout
# TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的
# scale，所以用Dropout的时候可以不用考虑scale。

keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# Softmax，输出的是概率
# 添加一个 Softmax 输出层，这里我们需要将 1024 维转为 10 维，所以需要声明一个 [1024, 10] 的权重和 [10] 的偏置：
w_fc2 = weight([1024, 10])
b_fc1 = bias([10])
y = tf.nn.softmax(tf.matmul(h_fc1_dropout, w_fc2) + b_fc1)

# Loss
# Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳
# 计算交差熵
cross_entropy = -tf.reduce_sum(y_label * tf.log(y))
train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Prediction
correct_prediction = tf.equal(tf.argmax(y_label, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(total_steps + 1):
        batch = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch[0], y_label: batch[1], keep_prob: dropout_keep_prob})
        # Train accuracy
        if step % steps_per_test == 0:
            print('Training Accuracy', step,
                  sess.run(accuracy, feed_dict={x: batch[0], y_label: batch[1], keep_prob: 1}))

# Final Test
print('Test Accuracy', sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels,
                                                     keep_prob: 1}))
