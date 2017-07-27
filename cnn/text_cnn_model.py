# -*- coding: utf-8 -*-
# TextCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer.
# print("started...")
import tensorflow as tf
import numpy as np


def cnn_instantiate_weights(embed_shape, w_shape, b_shape, name_scope):
    """define all weights here"""
    with tf.name_scope(name_scope):  # embedding matrix
        # [self.vocab_size, self.embed_size]
        ts_embedding = tf.get_variable(
            "Embedding", shape=embed_shape, initializer=self.initializer)

def cnn_assemble(ts_logits_list):
    model_num = len(ts_logits_list)
    batch_size, class_num = ts_logits_list.shape.as_list()

    # [model_num]
    ts_w = tf.get_variable("W_projection", shape=[model_num]
                           initializer=tf.random_normal_initializer(stddev=0.1))
    ts_b = tf.constant(0.5)

    for i in model_num :
        # shape:[batch_size, class_num ,1]
        ts_logits_list[i]= tf.expand_dims(ts_logits_list[i], -1)

    # shape:[batch_size, class_num, model_num]
    ts_concat = tf.concat(ts_logits_list, 2)
    # shape:[None,num_classes]
    ts_logits = tf.matmul(ts_concat, ts_w) + ts_b
    return ts_logits


def cnn_inference(ts_embedding,
                  ts_input_x,
                  class_num,
                  filter_sizes=[3, 4, 5],
                  filter_num=64,
                  name_scope="cnn",
                  drop_keep_prob=0.5):
    """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
    # 0.=====>init variable
    vocab_size = ts_embedding.shape[0].value
    embedding_size = ts_embedding.shape[1].value
    batch_size = ts_input_x.shape[0].value
    sentence_size = ts_input_x.shape[1].value
    filter_total_num = len(filter_sizes) * filter_num

    with tf.name_scope(name_scope):
        # 1.=====>get emebedding of words in the sentence
        # [batch_size, setence_length, embed_size]
        ts_embedded = tf.nn.embedding_lookup(ts_embedding, ts_input_x)
        # [batch_size, setence_length, embed_size, 1]
        ts_embedded = tf.expand_dims(ts_embedded, -1)

        # 2.=====>loop each filter size
        ts_pooled_ouputs = []
        for i, filter_size in enumerate(filter_sizes):
            filter_shape = [filter_size, embedding_size, 1, filter_num]
            ts_filter = tf.get_variable(
                "filter_%s" % filter_size, filter_shape, initializer=initializer)
            # shape:[batch_size,sentence_size - filter_size + 1, 1, filter_num]
            ts_conv = tf.nn.conv2d(ts_embedded, ts_filter, strides=[
                                          1, 1, 1, 1], padding="VALID", name="conv")
            # bias
            ts_conv_b = tf.get_variable(
                "conv_b_%s" % filter_size, [filter_num])
            # shape:[batch_size,sentence_size- filter_size + 1,1,num_filters]
            ts_conv_h = tf.nn.relu(tf.nn.bias_add(
                ts_conv, ts_conv_b), "relu")

            pool_shape = [1, sentence_size - filter_size + 1, 1, 1]
            # shape:[batch_size, 1, 1, num_filters]
            ts_pooled = tf.nn.max_pool(h, ksize=pool_shape, strides=[
                                       1, 1, 1, 1], padding='VALID', name="pool")
            ts_pooled_ouputs.append(ts_pooled)

        # 3.=====>combine all pooled filters ouput
        # shape:[batch_size, 1, 1, num_filters_total]
        ts_concat = tf.concat(ts_pooled_ouputs, 3)
        # shape:[None,num_filters_total]
        ts_concat_flat = tf.reshape(
            ts_concat, [-1, filter_total_num])

        # 4.=====>add dropout: use tf.nn.dropout
        # [None,num_filters_total]
        ts_dropout = tf.nn.dropout(
            ts_concat_flat, keep_prob=dropout_keep_prob)

        # 5.=====>logits(use linear layer):w*h+b
        # [number_filters_total, num_classes]
        w_shape = [filter_total_num, class_num]
        ts_w = tf.get_variable("W_projection", shape=w_shape,
                               initializer=tf.random_normal_initializer(stddev=0.1))
        # [num_classes]
        b_shape = [class_num]
        ts_b = tf.get_variable("b_projection", shape=b_shape)
        # shape:[None,num_classes]
        ts_logits = tf.matmul(ts_dropout, ts_w) + ts_b
        return ts_logits


class TextCNN:
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, vocab_size, embed_size,
                 is_training, initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False, clip_gradients=5.0, decay_rate_big=0.50):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(
            learning_rate, trainable=False, name="learning_rate")  # ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(
            self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = initializer
        # how many filters totally.
        self.num_filters_total = self.num_filters * len(filter_sizes)
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients

        # add placeholder (X,label)
        self.input_x = tf.placeholder(
            tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(
            tf.int32, [None, ], name="input_y")  # y:[None,num_classes]
        # y:[None,num_classes]. this is for multi-label classification only.
        self.input_y_multilabel = tf.placeholder(
            tf.float32, [None, self.num_classes], name="input_y_multilabel")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(
            self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        # [None, self.label_size]. main computation graph is here.
        self.logits = self.inference()
        if not is_training:
            return
        if multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(
            self.logits, 1, name="predictions")  # shape:[None,]

        if not self.multi_label_flag:
            # tf.argmax(self.logits, 1)-->[batch_size]
            correct_prediction = tf.equal(
                tf.cast(self.predictions, tf.int32), self.input_y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32), name="Accuracy")  # shape=()
        else:
            # fuke accuracy. (you can calcuate accuracy outside of graph using
            # method calculate_accuracy(...) in train.py)
            self.accuracy = tf.constant(0.5)

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.Embedding = tf.get_variable("Embedding", shape=[
                                             self.vocab_size, self.embed_size], initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection", shape=[
                                                self.num_filters_total, self.num_classes], initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable(
                "b_projection", shape=[self.num_classes])  # [label_size] #ADD 2017.06.09

    def inference(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(
            self.Embedding, self.input_x)  # [None,sentence_length,embed_size]
        # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
        self.sentence_embeddings_expanded = tf.expand_dims(
            self.embedded_words, -1)

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is
        # 4-d. feature is a new variable
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                # ====>a.create filter
                filter = tf.get_variable("filter-%s" % filter_size, [
                                         filter_size, self.embed_size, 1, self.num_filters], initializer=self.initializer)
                # ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                # Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                # Conv.Returns: A `Tensor`. Has the same type as `input`.
                #         A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
                # 1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                # input data format:NHWC:[batch, height, width,
                # channels];output:4-D
                # shape:[batch_size,sequence_length - filter_size +
                # 1,1,num_filters]
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[
                                    1, 1, 1, 1], padding="VALID", name="conv")
                # ====>c. apply nolinearity
                b = tf.get_variable("b-%s" % filter_size,
                                    [self.num_filters])  # ADD 2017-06-09
                # shape:[batch_size,sequence_length - filter_size +
                # 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                #                  ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                # strides: A list of ints that has length >= 4.  The stride of
                # the sliding window for each dimension of the input tensor.
                # shape:[batch_size, 1, 1, num_filters].max_pool:performs the
                # max pooling on the input.
                pooled = tf.nn.max_pool(h, ksize=[
                                        1, self.sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates
        # tensors along one dimension.where
        # num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool = tf.concat(pooled_outputs, 3)
        # shape should be:[None,num_filters_total]. here this operation has
        # some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) &
        # (3, 3)---->(1,9)
        self.h_pool_flat = tf.reshape(
            self.h_pool, [-1, self.num_filters_total])

        # 4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            # [None,num_filters_total]
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            # shape:[None,
            # self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
            logits = tf.matmul(
                self.h_drop, self.W_projection) + self.b_projection
        return logits

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as
            # `logits` with the softmax cross entropy loss.
            # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.input_y, logits=self.logits)
            # print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses)
            # # shape=(?,)
            # print("2.loss.loss:", loss) #shape=()
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(
                v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    # 0.0001#this loss function is for multi-label classification
    def loss_multilabel(self, l2_lambda=0.00001):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            # input_y:shape=(?, 1999); logits:shape=(?, 1999)
            # let `x = logits`, `z = labels`.  The logistic loss is:z *
            # -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.input_y_multilabel, logits=self.logits)
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
            # shape=(?, 1999).
            print("sigmoid_cross_entropy_with_logits.losses:", losses)
            # shape=(?,). loss for all data in the batch
            losses = tf.reduce_sum(losses, axis=1)
            # shape=().   average loss in the batch
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(
                v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
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
