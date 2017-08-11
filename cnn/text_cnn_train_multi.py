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

from text_cnn_model import TextCNN

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../data_util")

from data_util_zhihu import load_data_simple, create_voabulary_input, create_voabulary_output
from data_util_zhihu import pad_sequences
import cnn.text_cnn_util as cnn_util

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


class Options(object):
    """Options used by our tensorflow model."""

    def __init__(self):
        # Training options.
        self.batch_size = FLAGS.batch_size

        self.sequence_len = FLAGS.sentence_len
        self.char_vocab_size = 0
        self.word_vocab_size = 0
        self.char_embed_size = FLAGS.char_embed_size
        self.word_embed_size = FLAGS.word_embed_size

        # cnn filters
        self.filter_sizes = [3, 4, 5, 7]
        self.filter_num = FLAGS.num_filters

        # class num of labels
        self.class_num = FLAGS.num_classes

        # learning rate
        self.learning_rate = FLAGS.learning_rate
        self.decay_steps = FLAGS.decay_steps
        self.decay_rate = FLAGS.decay_rate

        self.clip_gradients = 5.0


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
    train_X[0] = convert2index(x_word2index, train_X[0])
    train_X[1] = convert2index(x_char2index, train_X[1])
    train_X[2] = convert2index(x_word2index, train_X[2])
    train_X[3] = convert2index(x_char2index, train_X[3])
    train_Y = convert2index(y_label2index, train_Y)

    # create embedding matrix
    char_embedding, word_embedding = None, None
    if use_static_embedding:
        word_embedding = get_embedding(x_word2index, zhihu_config['word_embedding'])
        char_embedding = get_embedding(x_char2index, zhihu_config['char_embedding'])

    # padding data
    paddings=zhihu_config['question_paddings_same']
    for i, X in enumerate(train_X):
        X = pad_sequences(X, maxlen=paddings[i], values=0.)

    # split train and test
    train_X, train_Y, testX, testY = split_train_and_test(train_X, train_Y)

    #########################################################################
    # 2. train process
    #########################################################################
    # create options
    options = Options()
    options.char_vocab_size = len(x_char2index)
    options.word_vocab_size = len(x_word2index)
    options.char_embed_size = len(char_embedding[0]) if char_embedding
    options.word_embed_size = len(word_embedding[0]) if word_embedding
    options.class_num = len(y_label2index)

    # create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        text_cnn = TextCNN(options, is_trainning=True)

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
                    sess, text_cnn, char_embedding, word_embedding)

        curr_epoch = sess.run(textCNN.epoch_step)

        # 3.feed data & training
        n = len(train_X[0])
        batch_size = FLAGS.batch_size

        for epoch in range(curr_epoch, FLAGS.num_epochs):
            counter = 0
            for start, end in zip(range(0, n, batch_size),
                                  range(batch_size, n, batch_size)):

                feed_dict = {text_cnn.title_char_X: train_X[0][start:end],
                             text_cnn.title_word_X: train_X[1][start:end],
                             text_cnn.desc_char_X: train_X[2][start:end],
                             text_cnn.desc_word_X: train_X[3][start:end],
                             text_cnn.input_y: train_Y[start:end],
                             text_cnn.dropout_keep_prob: 0.5}

                train_loss, train_acc, _ = sess.run([text_cnn.loss_val,
                                                     text_cnn.accuracy,
                                                     text_cnn.train_op],
                                                    feed_dict)
                counter += 1

                if counter % 50 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" % (
                        epoch, counter, train_loss, train_acc))

            # epoch increment
            sess.run(text_cnn.epoch_increment)

            # 4. test_set valid and save model
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(
                    sess, text_cnn, testX, testY, batch_size)

                print("Epoch %d Test Loss:%.3f\tTest Accuracy: %.3f" % (
                    epoch, eval_loss, eval_acc))

                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)

        # 5. valid model after training
        valid_loss, valid_acc = do_eval(
            sess, text_cnn, valid_X, valid_Y, batch_size)

        print("Finish training!!!Validation Loss:%.3f\tValidation Accuracy: %.3f" % (
            valid_loss, valid_acc))

    pass


def assign_pretrained_embedding(sess, text_cnn, char_embedding, word_embedding):

    tensor_word_embedding = tf.constant(
        word_embedding, dtype=tf.float32)  # convert to tensor
    tensor_char_embedding = tf.constant(
        char_embedding, dtype=tf.float32)  # convert to tensor

    # assign this value to our embedding variables of our model.
    assign_char_embedding = tf.assign(text_cnn.char_embedding,
                                      tensor_char_embedding)
    assign_word_embedding = tf.assign(text_cnn.word_embedding,
                                      tensor_word_embedding)

    sess.run([assign_char_embedding, assign_word_embedding])


def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, text_cnn, word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:",
          word2vec_model_path)
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector
    # fiel.word2vecc['w91874']
    word2vec_model = word2vec.load(word2vec_model_path)
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    # create an empty word_embedding list.
    word_embedding_2dlist = [[]] * vocab_size
    # assign empty for first word:'PAD'
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            # try to get vector:it is an array.
            embedding = word2vec_dict[word]
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(
                -bound, bound, FLAGS.embed_size)
            # init a random value for the word.
            count_not_exist = count_not_exist + 1
    # covert to 2d array.
    word_embedding_final = np.array(word_embedding_2dlist)
    word_embedding = tf.constant(
        word_embedding_final, dtype=tf.float32)  # convert to tensor
    # assign this value to our embedding variables of our model.
    t_assign_embedding = tf.assign(text_cnn.Embedding, word_embedding)
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist,
          " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


def do_eval(sess, text_cnn, eval_X, eval_Y, batch_size):
    eval_loss, eval_acc, counter = 0.0, 0.0, 0.0
    for start, end in zip(range(0, len(eval_X), batch_size),
                          range(batch_size, len(eval_X), batch_zie)):

        feed_dict = {text_cnn.title_char_X: eval_X[0][start:end],
                     text_cnn.title_word_X: eval_X[1][start:end],
                     text_cnn.desc_char_X: eval_X[2][start:end],
                     text_cnn.desc_word_X: eval_X[3][start:end],
                     text_cnn.input_y: eval_Y[start:end],
                     text_cnn.dropout_keep_prob: 0.5}
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
