import tensorflow as tf
import lib
import record_make
from PIL import Image
import numpy as np

class CLR:
    def __init__(self):
        # default parameters
        self.w_conv1 = lib.weight_variable('cls_w_conv1', [3, 3, 1, 64])
        self.b_conv1 = lib.bias_variable('cls_b_conv1', [64])
        self.w_conv2 = lib.weight_variable('cls_w_conv2', [3, 3, 64, 256])
        self.b_conv2 = lib.bias_variable('cls_b_conv2', [256])
        self.w_fc1 = lib.weight_variable('cls_w_fc1', [7 * 7 * 256, 1024])
        self.b_fc1 = lib.bias_variable('cls_b_fc1', [1024])
        self.w_fc2 = lib.weight_variable('cls_w_fc2', [1024, 10])
        self.b_fc2 = lib.bias_variable('cls_b_fc2', [10])

    def build(self):
        x = tf.placeholder(tf.float32, [None, 784])
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])
        right_label = tf.placeholder(tf.float32, [None, 10])
        # is_train = tf.placeholder('bool')
        classifier_loss = self.classifier(x_reshape)
        return x, classifier_loss, right_label

    def classifier(self, x):
        with tf.name_scope("classifier"):
            # conv_1 layer+pooling_1
            output = lib.conv2d(x, self.w_conv1) + self.b_conv1
            output = lib.batchnormalize(output)
            output = lib.pooling(tf.nn.relu(output))  # input: NWHC N*28*28*1 output: N*14*14*64

            # conv2 layer+pooling_2
            output = lib.conv2d(output, self.w_conv2) + self.b_conv2
            output = lib.batchnormalize(output)
            output = lib.pooling(tf.nn.relu(output))  # input: NWHC N*14*14*64 output: N*7*7*256

            # layer 3:full connect
            output = tf.reshape(output, [-1, 7 * 7 * 256])
            output = tf.matmul(output, self.w_fc1) + self.b_fc1
            output = lib.batchnormalize(output)
            output = tf.nn.relu(output)  # output shape: W*1024

            # layer 4:softmax layer   input:W*1024 output: 10-one-hat vector
            output = tf.matmul(output, self.w_fc2) + self.b_fc2
            output = tf.nn.softmax(output)
            return output

    def tstclassifier(self):
        input_t = tf.placeholder(tf.float32, shape=[28, 28])
        inter_t = tf.reshape(input_t, [1, 28, 28, 1])
        prob_index = self.classifier(inter_t)
        return prob_index, input_t



def train(model, batch_size):
    x, classifier_loss, right_label = model.build()
    cls_vars = [var for var in tf.trainable_variables() if "cls" in var.name]
    sess = tf.InteractiveSession()
    learning_rate = 0.0005
    iteration = 4100

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=classifier_loss, labels=right_label)
    correct_prediction = tf.equal(tf.argmax(classifier_loss, 1), tf.argmax(right_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, var_list=cls_vars)     # train operation
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=2)
    for i in range(iteration):
        batch = lib.mnist.train.next_batch(batch_size)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], right_label: batch[1]})
            print(" :step: " + str(i) + " training accuracy: " + str(train_accuracy))
        train_step.run(feed_dict={x: batch[0], right_label: batch[1]})
        #if i % 8000 == 0:
            #saver.save(sess, "./classifier-models/model.ckpt", global_step=i)
    print("test accuracy %g" % accuracy.eval(feed_dict={
            x: lib.mnist.test.images, right_label: lib.mnist.test.labels}))

    #sess.close()


#def tst(model, data_size, model_path="./classifier-models/model.ckpt"):
    #x, cls, _ = model.build()
    filename_queue = tf.train.string_input_producer(["./GAN_mnist_test.tfrecords"]) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [1, 784])
    label = tf.cast(features['label'], tf.int32)
    #sess = tf.InteractiveSession()
    #sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #saver = tf.train.Saver(max_to_keep=2)
    #saver.restore(sess, model_path)
    #print("model reload")
    k_index = tf.argmax(classifier_loss, 1)
    for i in range(3):
        img_xs, _ = sess.run([image, label])
        img_x = img_xs/255
        k = k_index.eval(feed_dict={x: img_xs})
        g = np.reshape(img_xs, (28, 28))

        print('k is')
        print(k)
       # print('k is %d' %(k))
        g = Image.fromarray(g, 'L')
        g.show()
        lib.mkdir('./reslut/%d/' %(k))
        g.save('./reslut/%d/%d.jpg' %(k, i))
    print("Test finished")
    coord.request_stop()
    coord.join(threads)
    sess.close()

