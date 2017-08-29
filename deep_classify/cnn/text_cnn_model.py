# -*- coding: utf-8 -*-
# TextCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer.
# print("started...")
import tensorflow as tf
import numpy as np


class TextCNN:
    def copy_options(options):
        # Training options.
        self.batch_size = options.batch_size

        self.sequence_len = options.sentence_len

        self.vocab_size = options.vocab_size
        self.embed_size = options.embed_size

        # cnn filters
        self.filter_sizes = options.filter_sizes
        self.filter_num = options.filter_num

        # class num of labels
        self.class_num = options.class_num

        # learning rate
        self.learning_rate = options.learning_rate
        self.decay_steps = options.decay_steps
        self.decay_rate = options.decay_rate

        self.clip_gradients = options.clip_gradients

        self.filter_total_num = options.filter_num * len(options.filter_sizes)

    def __init(self, options, is_traininig):

        # set hyperparamter
        self.copy_options(options)

        self.initializer = tf.random_normalinitializer(stddev=0.1)

        self.learning_rate = tf.Variable(
            self.learning_rate, trainable=False, name="learning_rate")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(
            self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        # add placeholder (X,label)
        self.input_x = tf.placeholder(
            tf.int32, [None, options.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        self.input_y_multilabel = tf.placeholder(
            tf.float32, [None, self.num_classes], name="input_y_multilabel")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        self.instantiate_weights()

        self.logits = self.inference()

        if not is_training:
            return

        self.loss_val = self.loss()

        self.train_op = self.train()

        self.predictions = tf.argmax(
            self.logits, 1, name="predictions")  # shape:[None,]

        correct_prediction = tf.equal(
            tf.cast(self.predictions, tf.int32), self.input_y)

        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name="Accuracy")

    def instantiate_weights(self):
        """define all weights here"""
        self = self.options
        with tf.name_scope("embedding"):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[
                                             self.vocab_size, self.embed_size], initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection", shape=[
                                                self.filter_total_num, self.class_num], initializer=self.initializer)
            self.b_projection = tf.get_variable(
                "b_projection", shape=[self.class_num])

    def inference(self):
        """main computation graph here: 1.embedding-->2.conv layer-->3.linear classifier"""
        self = self.options

        # 1.=====>get emebedding of words in the sentence
        # [batch_size, sentence_len, embed_size]
        self.embedded_words = tf.nn.embedding_lookup(
            self.Embedding, self.input_x)

        # [batch_size, sentence_len, embed_size, 1]
        self.sentence_embeddings_expanded = tf.expand_dims(
            self.embedded_words, -1)



        # 2.=====>conv-->hx+b-->max_pool layer
        pooled_outputs = []
        # NHWC
        height = self.sequence_length
        width = self.embed_size
        in_channels = 1
        out_channels = self.filter_num

        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                # ====>a.create conv kernel
                kernel = tf.get_variable("filter-%s" % filter_size, [
                                         filter_size, width, in_channels, out_channels], initializer=self.initializer)

                # ====>b.conv layer
                # Input Tensor Shape: [batch_size, height, width, in_chanel]
                # Output Tensor Shape: [batch_size, height-filter_size+1, 1, out_channel]
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, kernel, strides=[
                                    1, 1, 1, 1], padding="VALID", name="conv-%s" % filter_size)
                # ====>c. apply nolinearity
                b = tf.get_variable("b-%s" % filter_size, [out_channels])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")

                # ====>d. max_pool
                # [batch_size, 1, 1, out_channels]
                pooled = tf.nn.max_pool(h, ksize=[1, height - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        # [batch_size, 1, 1, filter_total_num]
        self.h_pool = tf.concat(pooled_outputs, 3)
        # [batch_size, filter_total_num]
        self.h_pool_flat = tf.reshape(
            self.h_pool, [-1, self.num_filters_total])

        # 4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            # [None,num_filters_total]
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        # 5.=====>logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            # shape:[None,self.num_classes]
            logits = tf.matmul(
                self.h_drop, self.W_projection) + self.b_projection
        return logits

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.input_y, logits=self.logits)

            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(
                v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda

            loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        self = self.options

        learning_rate = tf.train.exponential_decay(
            self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(
            self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op

# test started
# def test():
    # below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
    # num_classes=3
    # learning_rate=0.01
    # batch_size=8
    # decay_steps=1000
    # decay_rate=0.9
    # sequence_length=5
    # vocab_size=10000
    # embed_size=100
    # is_training=True
    # dropout_keep_prob=1 #0.5
    # filter_sizes=[3,4,5]
    # num_filters=128
    #textRNN=TextCNN(filter_sizes,num_filters,num_classes, learning_rate, batch_size, decay_steps, decay_rate,sequence_length,vocab_size,embed_size,is_training)
    # with tf.Session() as sess:
    #   sess.run(tf.global_variables_initializer())
    #   for i in range(100):
    #        input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
    #        input_x[input_x>0.5]=1
    #       input_x[input_x <= 0.5] = 0
    #       input_y=np.array([1,0,1,1,1,2,1,1])#np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
    #       loss,acc,predict,W_projection_value,_=sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.W_projection,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
    #       print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)
        # print("W_projection_value_:",W_projection_value)
# test()
