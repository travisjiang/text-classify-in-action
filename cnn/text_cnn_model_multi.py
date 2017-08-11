# -*- coding: utf-8 -*-
# TextCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer.
# print("started...")
import tensorflow as tf
import numpy as np


class TextCNN:
    def __init__(self, options, is_training):
        """init all hyperparameter here"""
        self._options = options
        self._initializer = tf.random_normal_initializer(stddev=0.1)
        self._is_training = is_training
        self._filter_total_num = options.filter_num * len(options.filter_sizes)

        # epoch step
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(
            self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        # add placeholder (X,label)
        add_placeholder()

        # initial weights
        self.instantiate_weights()

        # inference
        # [None, self.label_size]. main computation graph is here.
        self.logits = self.inference()

        if not is_training:
            return

        # loss
        print("going to use single label loss.")
        self.loss_val = self.loss()

        # train
        self.train_op = self.train()

        # prediction
        self.predictions = tf.argmax(
            self.logits, 1, name="predictions")  # shape:[None,]

        # accuracy
        correct_prediction = tf.equal(
            tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32), name="Accuracy")

    def add_placeholder(self):
        opts = self._options
        self.title_char_X = tf.placeholder(
            tf.int32, [None, opts.sequence_length], name="title_char_x")
        self.title_word_X = tf.placeholder(
            tf.int32, [None, opts.sequence_length], name="title_word_x")
        self.desc_char_X = tf.placeholder(
            tf.int32, [None, opts.sequence_length], name="desc_char_x")
        self.desc_word_X = tf.placeholder(
            tf.int32, [None, opts.sequence_length], name="desc_word_x")

        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

    def instantiate_weights(self):
        """define all weights here"""
        embedding_info = self._options.embedding_info

        embedding_map = {}
        with tf.name_scope("embedding"):
            for name, info in embedding_info:
                embedding_map[name] = tf.get_variable(
                    name, shape=info.shape, initializer=self._initializer)
            self._W_projection = tf.get_variable("W_projection",
                                                 shape=[self._filter_total_num,
                                                        self.num_classes],
                                                 initializer=self.initializer)
            self._b_projection = tf.get_variable("b_projection",
                                                 shape=[self.num_classes])

    def inference(self):
        """
        main computation graph here: 1.embedding-->2.average-->3.linear classifier
        """
        opts = self._options
        # 1.=====>get emebedding of words in the sentence
        # [batch_size, sentence_len, embed_size]
        for name, embedding in self.embedding_map:
            embedded_title_words = tf.nn.embedding_lookup(
                self.word_embedding, self.title_word_X)

        embedded_title_words = tf.nn.embedding_lookup(
            self.word_embedding, self.title_word_X)
        embedded_title_chars = tf.nn.embedding_lookup(
            self.char_embedding, self.title_char_X)
        embedded_desc_words = tf.nn.embedding_lookup(
            self.word_embedding, self.desc_word_X)
        embedded_desc_chars = tf.nn.embedding_lookup(
            self.char_embedding, self.desc_char_X)

        embedded_list = [embedded_title_words,
                         embedded_title_chars,
                         embedded_desc_words,
                         embedded_desc_chars]
        # [batch_size, sentence_len, embed_size, 1]
        embedded_list = [tf.expand_dims(e, -1) for e in embedded_list]
        # [batch_size, sentence_len, embed_size, in_channels(=4)]
        embeddes_x = tf.concat(embedded_list, 3)
        in_channels = 4
        out_channels = opts.filter_num

        pooled_outputs = []
        for i, filter_size in enumerate(opts.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                filter = tf.get_variable("filter-%s" % filter_size,
                                         [filter_size, opts.embed_size,
                                             in_channels, out_channels],
                                         initializer=self._initializer)
                # [batch_size, sentence_len-filter_size+1, 1, out_channels]
                conv = tf.nn.conv2d(embedded_x,
                                    filter, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")

                b = tf.get_variable("b-%s" % filter_size, [out_channels])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")

                # [batch_size, 1, 1, out_channels]
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.sequence_length -
                                               filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")

                pooled_outputs.append(pooled)

        # [batch_size, 1, 1, filter_total_num]
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(
            h_pool, [-1, self._filter_total_num])

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(
                h_pool_flat, keep_prob=self.dropout_keep_prob)

        with tf.name_scope("output"):
            logits = tf.matmul(
                h_drop, self.W_projection) + self.b_projection
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
        opts = self._options
        global_step = tf.Variable(0, trainable=False, name="global_step")

        learning_rate = tf.Variable(
            opts.learning_rate, trainable=False, name="learning_rate")

        learning_rate = tf.train.exponential_decay(
            learning_rate, global_step,
            opts.decay_steps, opts.decay_rate, staircase=True)

        train_op = tf.contrib.layers.optimize_loss(
            opts.loss_val,
            global_step=global_step,
            learning_rate=learning_rate,
            optimizer="Adam",
            clip_gradients=opts.clip_gradients)
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
