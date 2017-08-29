# -*- coding: utf-8 -*-
# training the model.
# process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed
# data. 4.training (5.validation) ,(6.prediction)
import sys
# sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from tflearn.data_utils import to_categorical  # , pad_sequences
import os
import word2vec
import pickle

from text_cnn_model_tdwc import TextCnnTDWC
from options import OptionsTDWC

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../data_util")

from data_util_zhihu import load_data, create_voabulary_from_data, create_voabulary_label
from data_util_zhihu import convert2index, get_embedding, pad_sequences

# configuration
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes", 215, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer(
    "batch_size", 512, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer(
    "decay_steps", 6000, "how many steps before decay learning rate.")  # 6000批处理的大小 32-->128
tf.app.flags.DEFINE_float(
    "decay_rate", 0.65, "Rate of decay for learning rate.")  # 0.65一次衰减多少
# tf.app.flags.DEFINE_integer("num_sampled",50,"number of noise sampling") #100
tf.app.flags.DEFINE_string(
    "ckpt_dir", "text_cnn_title_desc_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 50, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 256, "embedding size")
tf.app.flags.DEFINE_boolean(
    "is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 15, "number of epochs to run.")
tf.app.flags.DEFINE_integer(
    "validate_every", 3, "Validate every validate_every epochs.")  # 每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_static_embedding", True,
                            "whether to use embedding or not.")
tf.app.flags.DEFINE_string(
    "cache_path", "text_cnn_checkpoint/data_cache.pik", "checkpoint location for the model")
tf.app.flags.DEFINE_string(
    "training_data_path", "data_set/preprocessd_data/train_set_small_title.txt", "path of traning data.")
tf.app.flags.DEFINE_integer(
    "num_filters", 64, "number of filters")  # 256--->512
tf.app.flags.DEFINE_string(
    "word2vec_model_path", "data_set/ieee_zhihu_cup/word_embedding.txt", "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_boolean(
    "multi_label_flag", False, "use multi label or single label.")


def main(_):
    #########################################################################
    # 1. data process
    #########################################################################

    # load Data
    # train_X contains multi train X data
    # train_X = [title_char_X, title_word_X, desc_char_X, desc_word_X]
    train_X, train_Y = load_data(
        zhihu_config['train_set_question_topic'])

    valid_X, valid_Y = load_data(
        zhihu_config['valid_set_question_topic'])

    # create vocabulary
    special_words = ["_PAD"]
    if use_static_embedding:
        x_word2index, x_index2word = create_vocabulary_from_model(
            zhihu_config['word_embedding'], special_words)
        x_char2index, x_index2char = create_vocabulary_from_model(
            zhihu_config['char_embedding'], special_words)
    else:
        x_word2index, x_index2word = create_vocabulary_from_data(
            train_X[1] + train_X[3] + valid_X[1] + valid_X[3], special_words)
        x_char2index, x_index2char = create_vocabulary_from_data(
            train_X[0] + train_X[2] + valid_X[0] + valid_X[2], special_words)

    y_label2index, y_index2label = create_vocabulary_from_data(
        train_Y + valid_Y, special_words)

    # convert words-matrix to index-matrix
    def _convert2index_XY(X, Y):
        X[0] = convert2index(x_word2index, X[0])
        X[1] = convert2index(x_char2index, X[1])
        X[2] = convert2index(x_word2index, X[2])
        X[3] = convert2index(x_char2index, X[3])
        Y = convert2index(y_label2index, Y)

    train_X, train_Y = _convert2index_XY(train_X, train_Y)
    valid_X, valid_Y = _convert2index_XY(valid_X, valid_Y)

    # padding data
    paddings = zhihu_config['question_paddings_same']
    for i in len(train_X):
        train_X[i] = pad_sequences(train_X[i], maxlen=paddings[i], values=0.)

    for i in len(valid_X):
        valid_X[i] = pad_sequences(valid_X[i], maxlen=paddings[i], values=0.)

    # create embedding matrix
    char_embedding, word_embedding = None, None
    if use_static_embedding:
        word_embedding = get_embedding(
            x_word2index, zhihu_config['word_embedding'])
        char_embedding = get_embedding(
            x_char2index, zhihu_config['char_embedding'])
    char_embed_size = len(
        char_embedding[0]) if char_embedding else FLAGS.embed_size
    word_embed_size = len(
        word_embedding[0]) if word_embedding else FLAGS.embed_size

    # split train and test TODO
    train_X, train_Y, testX, testY = split_train_and_test(train_X, train_Y)

    #########################################################################
    # 2. train process
    #########################################################################
    # for each input_x, name:(sequence_length, input_x_matrix)
    # for this model, sequence_length must be same
    assert(paddings[0] == paddings[1] == paddings[2] == paddings[3])
    input_types = ["title_char", "title_word", "desc_char", "desc_word"]
    data_info = {input_types[0]: (paddings[0], train_X[0]),
                 input_types[1]: (paddings[1], train_X[1]),
                 input_types[2]: (paddings[2], train_X[2]),
                 input_types[3]: (paddings[3], train_X[3])}
    # for each embed of input_x, name:(vocab_size, embed_size, embedding_matrix)
    # for this model, embed_size must be same
    assert(char_embed_size == word_embed_size)
    embed_info = {input_types[0]: (len(x_char2index), char_embed_size, char_embedding),
                  input_types[1]: (len(x_word2index), word_embed_size, word_embedding),
                  input_types[2]: (len(x_char2index), char_embed_size, char_embedding),
                  input_types[3]: (len(x_word2index), word_embed_size, word_embedding)}
    # create options
    options = OptionsTDWC()
    options.input_type_num = 4
    options.data_info = dict(data_info.items()[0:4])
    options.embed_info = dict(embed_info.items()[0:4])
    options.sequence_length = paddings[0]
    options.embed_size = char_embed_size
    options.class_num = len(y_label2index)

    # create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        text_cnn = TextCnnTDWC(options, is_trainning=True)

        # Initialize Save
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint...")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables...')
            sess.run(tf.global_variables_initializer())
            print('Loading pretrained word embedding...')
            if FLAGS.use_static_embedding:
                assign_pretrained_embedding(
                    sess, text_cnn, embed_info)

        curr_epoch = sess.run(text_cnn.epoch_step)

        # 3.feed data & training
        n = len(train_X[0])
        batch_size = FLAGS.batch_size

        for epoch in range(curr_epoch, FLAGS.num_epochs):
            total_loss, total_acc, counter = 0.0, 0.0, 0.0
            for start, end in zip(range(0, n, batch_size),
                                  range(batch_size, n, batch_size)):

                feed_dict = {text_cnn.input_y: train_Y[start:end],
                             text_cnn.dropout_keep_prob: 0.5}

                for name, input_x in text_cnn.input_x_map:
                    feed_dict[input_x] = data_info[name][start:end]

                train_loss, train_acc, _ = sess.run([text_cnn.loss_val,
                                                     text_cnn.accuracy,
                                                     text_cnn.train_op],
                                                    feed_dict)
                total_loss += train_loss
                total_acc += train_acc
                counter += 1

                if counter % 50 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" % (
                        epoch, counter, total_loss / counter, total_acc / counter))

            # epoch increment
            sess.run(text_cnn.epoch_increment)

            # 4. test_set valid and save model
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(
                    sess, text_cnn, testX, testY, batch_size, input_types)

                print("Epoch %d Test Loss:%.3f\tTest Accuracy: %.3f" % (
                    epoch, eval_loss, eval_acc))

                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)

        # 5. valid model after training
        valid_loss, valid_acc = do_eval(
            sess, text_cnn, valid_X, valid_Y, batch_size, input_types)

        print("Finish training!!!Validation Loss:%.3f\tValidation Accuracy: %.3f" % (
            valid_loss, valid_acc))

    pass


def assign_pretrained_embedding(sess, text_cnn, embed_info):

    assign_ops = []
    for name, embed_tensor in text_cnn.embed_map.items():
        # embedding_info: {name: (vocab_size, embed_size, embed_matrix)}
        pretrained_embedding = tf.constant(
            embed_info[name][2], dtype=tf.float32)

        # assign this value to our embedding variables of our model.
        assign_embedding = tf.assign(embed_tensor, pretrained_embedding)
        assign_ops.append(assign_embedding)

    sess.run(assign_ops)


def do_eval(sess, text_cnn, eval_X, eval_Y, batch_size, input_types):
    eval_loss, eval_acc, counter = 0.0, 0.0, 0.0
    for start, end in zip(range(0, len(eval_X), batch_size),
                          range(batch_size, len(eval_X), batch_zie)):

         feed_dict = {text_cnn.input_y: eval_Y[start:end],
                 text_cnn.dropout_keep_prob: 0.5}

        for name, input_x in text_cnn.input_x_map:
            feed_dict[input_x] = eval_X[input_types.index(name)][start:end]

        # result of this batch
        loss, acc, logits = sess.run([text_cnn.loss_val,
                                      text_cnn.accuracy,
                                      text_cnn.logits])

        eval_loss += loss
        eval_acc += acc
        counter += 1.0

        return eval_loss / counter, eval_acc / counter


if __name__ == "__main__":
    tf.app.run()
