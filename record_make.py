import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def save_record():
    cwd = './tf.record'
    classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    writer = tf.python_io.TFRecordWriter("./GAN_mnist_test.tfrecords")  # the file path of tf.record

    for index, name in enumerate(classes):
        class_path = cwd + '/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((28, 28))
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())

    writer.close()


def read_record():
    filename_queue = tf.train.string_input_producer(["./GAN_mnist_test.tfrecords"]) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [28, 28])
    label = tf.cast(features['label'], tf.int32)
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(5):
        image_s, label_s = sess.run([image, label])
        print(image_s)
        image_s = np.reshape(image_s, (28, 28))
        picture = Image.fromarray(image_s, 'L')
        picture.show()
    coord.request_stop()
    coord.join(threads)
    sess.close()

