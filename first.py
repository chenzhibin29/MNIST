# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入数据集
# Get MNIST Data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 定义一些超参数
# Variables
# 随机抓取训练数据中的 batch_size 个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行 train，seesion会话里。
batch_size = 100
total_steps = 5000  # 训练次数
steps_per_test = 100    # 每100次打印一次准确率

# 构建模型
# Build Model
x = tf.placeholder(tf.float32, [None, 784])  # placeholder() 方法声明即可，一会我们在运行的时候传递给它真实的数据就好，第一个参数是数据类型，第二个参数是形状
y_label = tf.placeholder(tf.float32, [None, 10])
w = tf.Variable(tf.zeros([784, 10]))  # Variable 初始化了 TensorFlow 中的变量，b 初始化为一个常量，w 是一个随机初始化的 1×2 的向量，范围在 -1 和 1 之间，
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)
# matmul() 方法就是 TensorFlow 中提供的矩阵乘法，类似 Numpy 的 dot() 方法。不过不同的是 matmul() 不支持向量和矩阵相乘，即不能 BroadCast，
# 所以在这里做乘法前需要先调用 reshape() 一下转成 1×2 的标准矩阵，最后将结果表示为 y。
# softmax 函数会将求出来的值转换为一个概率值。

# 损失函数，学习速率为0.5
# Loss
# 计算交叉熵
# 首先用 reduce_sum() 方法针对每一个维度进行求和，reduction_indices 是指定沿哪些维度进行求和。
# 然后调用 reduce_mean() 则求平均值，将一个向量中的所有元素求算平均值。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y), reduction_indices=[1]))
# 梯度下降法，GradientDescentOptimizer(0.5)
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# tf.argmax() 是一个非常有用的函数，它能给出某个 Tensor 对象在某一维上的其数据最大值所在的索引值
# 由于标签向量是由 0,1 组成，因此最大值 1 所在的索引位置就是类别标签，
# 比如 tf.argmax(y, 1) 返回的是模型对于任一输入 x 预测到的标签值，而 tf.argmax(y_label, 1) 代表正确的标签，
# 我们可以用 tf.equal() 方法来检测我们的预测是否真实标签匹配（索引位置一样表示匹配）。
# Prediction


'''
tf.equal(a, b) 
此函数比较等维度的a, b矩阵相应位置的元素是否相等，相等返回True,否则为False 
返回：同维度的矩阵，元素值为True或False 
tf.cast(x, dtype, name=None) 
将x的数据格式转化成dtype.例如，原来x的数据格式是bool， 
那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以
tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None) 
功能：求某维度的均值 
'''


correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_label, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 最后我们在session里运行这个模型
# Session会话是tensorflow里面的重要机制，tensorflow构建的计算图必须通过Session会话才能执行，
# 如果只是在计算图中定义了图的节点但没有使用Session会话的话，就不能运行该节点。
# 调用session的run方法
# Run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Train 10000 steps
    for step in range(total_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={x: batch_x, y_label: batch_y})
        # Test every 100 steps
        # 计算测试数据集上的正确率
        if step % steps_per_test == 0:
            print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))


# tf.Variable类创建了一个具有可以修改的初始值的张量，很像一个普通的Python变量。 该张量在会话中存储其状态，因此您必须手动初始化张量的状态。
# 使用session(会话)调用tf.global_variables_initializer()操作。tf.global_variables_initializer()会返回一个操作，从计算图中初始化所有TensorFlow变量。

'''
tf.nn.max_pool(value, ksize, strides, padding, name=None) 
参数是四个，和卷积很类似： 
第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map， 
    依然是[batch, height, width, channels]这样的shape 
第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们 
    不想在batch和channels上做池化，所以这两个维度设为了1 
第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1] 
第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME' 
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式 
'''