import tensorflow as tf
import numpy
import scipy.misc
import scipy.io
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './data', 'Directory for storing data') # 第一次启动会下载文本资料

print(FLAGS.data_dir)
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


def weight_variable(name,shape):
    return tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def pooling(x):
    pooling_method = 'max'
    if pooling_method == 'avg':
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    else:
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0, 1, 2])
        std = tf.reduce_mean(tf.square(X - mean), [0, 1, 2])
        X = (X - mean) / tf.sqrt(std + eps)
        if g is not None and b is not None:
            g = tf.reshape(g, [1, 1, 1, -1])
            b = tf.reshape(b, [1, 1, 1, -1])
            X = X * g + b
    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X - mean), 0)
        X = (X - mean) / tf.sqrt(std + eps)
        if g is not None and b is not None:
            g = tf.reshape(g, [1, -1])
            b = tf.reshape(b, [1, -1])
            X = X * g + b
    else:
        raise NotImplementedError
    return X


def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


def save_visualization(X, nh_nw, save_path="./wdcgan-vis/sample.jpg"):
    h, w = X.shape[1], X.shape[2]
    img = numpy.zeros((h * nh_nw[0], w * nh_nw[1], X.shape[3]))
    for n, x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = x
    if X.shape[3] == 1:
        scipy.misc.imsave(save_path, img[:, :, 0])
    else:
        scipy.misc.imsave(save_path, img)


def save_dataset(X, nh_nw, save_path="./wdcgan-vis/sample.jpg"):
    h, w = X.shape[1], X.shape[2]
    img = numpy.zeros((h * nh_nw[0], w * nh_nw[1], X.shape[3]))
    for n, x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = x
    if X.shape[3] == 1:
        scipy.misc.imsave(save_path, img[0:h, 0:w, 0])
    else:
        scipy.misc.imsave(save_path, img)


def get_data(filename="data.mat"):
    try:
        datafile = scipy.io.loadmat(filename)
        data = datafile.get('data', None)
        assert data is not None
        data = data.astype(numpy.float32) / 255.
        label = datafile.get('label', None)
        assert label is not None
        return data, label
    except FileNotFoundError:
        print("No such file or directory:", filename)
        exit(2)


def OneHot(X, n=None, negative_class=0.):
    X = numpy.asarray(X).flatten()
    if n is None:
        n = numpy.max(X) + 1
    Xoh = numpy.ones((len(X), n)) * negative_class
    Xoh[numpy.arange(len(X)), X] = 1.
    return Xoh


def batch_norm(self, input, is_training):
    input_shape = input.get_shape()
    axis = list(range(len(input_shape) - 1))
    shape = input_shape[-1:]
    gamma = tf.Variable(tf.ones(shape), name='gamma')
    beta = tf.Variable(tf.zeros(shape), name='beta')
    moving_mean = tf.Variable(tf.zeros(shape), name='moving_mean',
                              trainable=False)
    moving_variance = tf.Variable(tf.ones(shape),
                                  name='moving_variance',
                                  trainable=False)
    control_inputs = []

    def f1():
        return 1

    def f0():
        return 0

    flag = tf.cond(is_training, f1, f0)
    if flag == 1:
        mean, variance = tf.nn.moments(input, axis)
        update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, self.decay)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, self.decay)
        control_inputs = [update_moving_mean, update_moving_variance]
    else:
        mean = moving_mean
        variance = moving_variance
    with tf.control_dependencies(control_inputs):
        return tf.nn.batch_normalization(
            input, mean=mean, variance=variance, offset=beta,
            scale=gamma, variance_epsilon=0.001)


def mkdir(path):
    import os

    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)

        #print(path + ' 创建成功')
        return True
    else:
        #print(path + ' 目录已存在')
        return False

