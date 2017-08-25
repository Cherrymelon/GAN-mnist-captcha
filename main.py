# coding:utf-8
import lib
import tensorflow as tf
import numpy as np
import time
import classifier
import record_make


class GAN_MODEL:
    def __init__(self, shape=(28, 28), channel=1, dim_z=128):
        self.shape = shape
        self.channel = channel
        self.dim_z = dim_z

        # discriminator
        self.dim_dis_1 = 64
        self.dim_dis_2 = 128
        self.dim_dis_3 = 1024
        self.dis_W1 = lib.weight_variable(name="dis_W1", shape=[5, 5, self.channel, self.dim_dis_1])
        self.dis_W2 = lib.weight_variable(name="dis_W2", shape=[5, 5, self.dim_dis_1, self.dim_dis_2])
        self.dis_W3 = lib.weight_variable(name="dis_W3", shape=[self.dim_dis_2 * (shape[0]//4) * (shape[1]//4), self.dim_dis_3])
        self.dis_W4 = lib.weight_variable(name="dis_W4", shape=[self.dim_dis_3, 1])

        # generator
        self.dim_gen_1 = 1024
        self.dim_gen_2 = 128
        self.dim_gen_3 = 64
        self.gen_W1 = lib.weight_variable(name="gen_W1", shape=[self.dim_z, self.dim_gen_1], )
        self.gen_W2 = lib.weight_variable(name="gen_W2", shape=[self.dim_gen_1, self.dim_gen_2 * (shape[0]//4) * (shape[1]//4)])
        self.gen_W3 = lib.weight_variable(name="gen_W3", shape=[5, 5, self.dim_gen_3, self.dim_gen_2])
        self.gen_W4 = lib.weight_variable(name="gen_W4", shape=[5, 5, self.channel, self.dim_gen_3])

    def build_model(self, batch_size):
        x = tf.placeholder(tf.float32, [None, self.shape[0] * self.shape[1]])
        z = tf.placeholder(tf.float32, [None, self.dim_z])
        real_image = tf.reshape(x, [-1, self.shape[0], self.shape[1], self.channel])
        fake_image = self.generate(batch_size, z)
        real = self.discriminate(real_image)
        fake = self.discriminate(fake_image)
        dis_loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
        gen_loss = -tf.reduce_mean(fake)
        alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = alpha * real_image + (1. - alpha) * fake_image
        gradients = tf.gradients(self.discriminate(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), 1))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        LAMBDA = 10
        dis_loss += LAMBDA * gradient_penalty
        return x, z, dis_loss, gen_loss

    def discriminate(self, image):
        with tf.name_scope("discriminator"):
            dis_output = tf.nn.conv2d(image, self.dis_W1, strides=[1, 2, 2, 1], padding="SAME") #
            dis_output = lib.lrelu(dis_output)
            dis_output = tf.nn.conv2d(dis_output, self.dis_W2, strides=[1, 2, 2, 1], padding="SAME")
            dis_output = lib.lrelu(dis_output)
            dis_output = tf.reshape(dis_output, [-1, self.dim_dis_2 * (self.shape[0] // 4) * (self.shape[1] // 4)])
            dis_output = tf.matmul(dis_output, self.dis_W3)
            dis_output = lib.batchnormalize(dis_output)
            dis_output = lib.lrelu(dis_output)
            dis_output = tf.matmul(dis_output, self.dis_W4)
            return dis_output

    def generate(self, batch_size, z):
        with tf.name_scope("generator"):
            gen_output = tf.matmul(z, self.gen_W1)
            gen_output = lib.batchnormalize(gen_output)
            gen_output = tf.nn.relu(gen_output)
            gen_output = tf.matmul(gen_output, self.gen_W2)
            gen_output = lib.batchnormalize(gen_output)
            gen_output = tf.nn.relu(gen_output)
            gen_output = tf.reshape(gen_output, [-1, self.shape[0] // 4, self.shape[1] // 4, self.dim_gen_2])
            gen_output = tf.nn.conv2d_transpose(gen_output, self.gen_W3, [batch_size, self.shape[0] // 2, self.shape[1] // 2, self.dim_gen_3], strides=[1, 2, 2, 1])
            gen_output = lib.batchnormalize(gen_output)
            gen_output = tf.nn.relu(gen_output)
            gen_output = tf.nn.conv2d_transpose(gen_output, self.gen_W4, [batch_size, self.shape[0], self.shape[1], self.channel], strides=[1, 2, 2, 1])
            gen_output = tf.nn.sigmoid(gen_output)
            return gen_output

    def samples_generator(self, batch_size):
        z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        image = self.generate(batch_size, z)
        return z, image


def train(model, batch_size, dim_z):
    learning_rate = 0.001
    n_epochs = 4000
    n_dis = 1
    n_gen = 5
    x, z, dis_loss, gen_loss = model.build_model(batch_size)

    dis_vars = [var for var in tf.trainable_variables() if "dis" in var.name]
    gen_vars = [var for var in tf.trainable_variables() if "gen" in var.name]

    train_op_dis = tf.train.RMSPropOptimizer(learning_rate).minimize(dis_loss, var_list=dis_vars)
    train_op_gen = tf.train.RMSPropOptimizer(learning_rate).minimize(gen_loss, var_list=gen_vars)

    sample_z, sample_image = model.samples_generator(batch_size=sampleSize[0] * sampleSize[1])
    tf.summary.scalar("dis_loss", dis_loss)
    tf.summary.scalar("gen_loss", gen_loss)
    merged_summary_op = tf.summary.merge_all()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10)
    summary_writer = tf.summary.FileWriter(".\wdcgan-logs", sess.graph)

    sample_z_input = np.random.uniform(-1, 1, size=(sampleSize[0] * sampleSize[1], dim_z)).astype(np.float32)

    start_time = time.clock()
    for iterations in range(n_epochs):
        dis_loss_val, gen_loss_val = 0.0, 0.0
        for _ in range(n_dis):
            input_x = lib.mnist.train.next_batch(batch_size)
            input_z = np.random.uniform(-1, 1, size=(batch_size, dim_z)).astype(np.float32)
            _, dis_loss_val = sess.run([train_op_dis, dis_loss], feed_dict={x: input_x[0], z: input_z})
        for _ in range(n_gen):
            input_z = np.random.uniform(-1, 1, size=(batch_size, dim_z)).astype(np.float32)
            _, gen_loss_val = sess.run([train_op_gen, gen_loss], feed_dict={z: input_z})
        if iterations % 20 == 0:
            print("Iter: %5d; dis_loss: %6f, gen_loss: %6f" % (iterations, dis_loss_val, gen_loss_val))
            samples = sess.run(sample_image, feed_dict={sample_z: sample_z_input})
            lib.save_visualization(samples, sampleSize,
                               save_path="./wdcgan-vis/sample_%06d_%09.4f.jpg" % (iterations, (time.clock() - start_time)))
            saver.save(sess, "./wdcgan-models/model.ckpt", global_step=iterations)
            input_x = lib.mnist.train.next_batch(batch_size)
            input_z = np.random.uniform(-1, 1, size=(batch_size, dim_z)).astype(np.float32)
            summary_writer.add_summary(sess.run(merged_summary_op, feed_dict={x: input_x[0], z: input_z}), iterations)
    summary_writer.close()
    sess.close()


def tst(model, sample_size, dim_z, model_path="models/model.ckpt"):
    sample_z, sample_image = model.samples_generator(batch_size=tst_pic_size[0] * tst_pic_size[1])

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10)
    saver.restore(sess, model_path)
    print("model restored")
    start_time = time.clock()
    for _ in range(output_number):
        sample_z_input = np.random.uniform(-1, 1, size=(tst_pic_size[0] * tst_pic_size[1], dim_z)).astype(np.float32)
        samples = sess.run(sample_image, feed_dict={sample_z: sample_z_input})
        lib.save_dataset(samples, tst_pic_size,
                           save_path="./tf.record/tst_%06d_%09.4f.jpg" %(output_number, (time.clock() - start_time)))
    sess.close()

if __name__ == "__main__":
    batchSize = 128
    imageShape = (28, 28, 1)
    imageChannel = 1
    dimZ = 128
    sampleSize = (16, 16)
    tst_pic_size = (2, 2)
    output_number = 5000
    dcgan_model = GAN_MODEL(imageShape, imageChannel, dimZ)
    #train(dcgan_model, batchSize, dimZ)
    #tst(dcgan_model, tst_pic_size, dimZ, model_path="./wdcgan-models/model.ckpt-%d" %(3940))
    #record_make.save_record()
    classifier_mod = classifier.CLR()
    classifier.train(classifier_mod, 100)
    #classifier.tst(classifier_mod, output_number, "./classifier-models/model.ckpt-8000")

