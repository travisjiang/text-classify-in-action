# -*- coding: utf-8 -*-
# training the model.
# process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed
# data. 4.training (5.validation) ,(6.prediction)
import sys
# sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from tflearn.data_utils import to_categorical#, pad_sequences
import os
import word2vec
import pickle

from text_cnn_model import TextCNN

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/../data_util")

from data_util_zhihu import load_data_simple, create_voabulary_input, create_voabulary_output
from data_util_zhihu import pad_sequences

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
tf.app.flags.DEFINE_boolean("use_embedding", True,
                            "whether to use embedding or not.")
tf.app.flags.DEFINE_string("cache_path","text_cnn_checkpoint/data_cache.pik","checkpoint location for the model")
tf.app.flags.DEFINE_string(
    "training_data_path", "data_set/preprocessd_data/train_set_small_title.txt", "path of traning data.")
tf.app.flags.DEFINE_integer(
    "num_filters", 64, "number of filters")  # 256--->512
tf.app.flags.DEFINE_string(
    "word2vec_model_path", "data_set/ieee_zhihu_cup/word_embedding.txt", "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_boolean(
    "multi_label_flag", False, "use multi label or single label.")
filter_sizes = [1, 2, 3, 4, 5, 6, 7]  # [1,2,3,4,5,6,7]
# 1.load data(X:list of lint,y:int). 2.create session. 3.feed data.
# 4.training (5.validation) ,(6.prediction)


def main(_):
    # 1. load data
    trainX, trainY, testX, testY = None, None, None, None
    X, Y, E = load_data(zhihu_config['train_set_question_topic'])
    # [None, sentence_len, embed_size]
    X_title_char, X_title_word, X_desc_char, X_desc_word = X
    X_char_embed, X_word_embed = E

    validX, validY = None, None
    X, Y, E = load_data(zhihu_config['valid_set_question_topic'])

    # 2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_and_validate(sess, trainX, trainY, validX, validY)
    pass

def train_and_validate():
        # Instantiate Model
        title_char_cnn = initialize_and_save(sess)
        title_word_cnn = initialize_and_save(sess)
        desc_char_cnn = initialize_and_save(sess)
        desc_word_cnn = initialize_and_save(sess)
        curr_epoch = sess.run(title_char_cnn.epoch_step)

        # 3.feed data & training
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                feed_dict = {title_char_cnn.input_x: X_title_char[start:end],
                        title_word_cnn.input_x: X_title_word[start:end],
                        desc_word_cnn.input_x: X_desc_word[start:end],

                        textCNN.dropout_keep_prob: 0.5}
                feed_dict[textCNN.input_y] = trainY[start:end]
                curr_loss, curr_acc, _ = sess.run(
                    [textCNN.loss_val, textCNN.accuracy, textCNN.train_op], feed_dict)  # curr_acc--->TextCNN.accuracy
                loss, counter, acc = loss + curr_loss, counter + 1, acc + curr_acc
                if counter % 50 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" % (
                        epoch, counter, loss / float(counter), acc / float(counter)))  # tTrain Accuracy:%.3f---》acc/float(counter)

            # epoch increment
            print("going to increment epoch counter....")
            sess.run(textCNN.epoch_increment)

            # 4. if need validation
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(
                    sess, textCNN, testX, testY, batch_size, vocabulary_index2word_label)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (
                    epoch, eval_loss, eval_acc))
                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)

        # 5. valid model after training
        test_loss, test_acc = do_eval(
            sess, textCNN, testX, testY, batch_size, vocabulary_index2word_label)

def initialize_and_save(sess):
    textCNN = TextCNN(filter_sizes, FLAGS.num_filters,
            FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size,
            FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sentence_len,
            vocab_size, FLAGS.embed_size, FLAGS.is_training,
            multi_label_flag=FLAGS.multi_label_flag)
    # Initialize Save
    saver = tf.train.Saver()
    if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
        print("Restoring Variables from Checkpoint")
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
    else:
        print('Initializing Variables')
        sess.run(tf.global_variables_initializer())
        if FLAGS.use_embedding:  # load pre-trained word embedding
            assign_pretrained_word_embedding(
                sess, vocabulary_index2word, vocab_size, textCNN, word2vec_model_path=FLAGS.word2vec_model_path)

    return textCNN

def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN, word2vec_model_path=None):
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
    t_assign_embedding = tf.assign(textCNN.Embedding, word_embedding)
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist,
          " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

# 在验证集上做验证，报告损失、精确度


def do_eval(sess, textCNN, evalX, evalY, batch_size, vocabulary_index2word_label):
    number_examples = len(evalX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        feed_dict = {
            textCNN.input_x: evalX[start:end], textCNN.dropout_keep_prob: 1}
        if not FLAGS.multi_label_flag:
            feed_dict[textCNN.input_y] = evalY[start:end]
        else:
            feed_dict[textCNN.input_y_multilabel] = evalY[start:end]
        curr_eval_loss, logits, curr_eval_acc = sess.run(
            [textCNN.loss_val, textCNN.logits, textCNN.accuracy], feed_dict)  # curr_eval_acc--->textCNN.accuracy
        #label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        #curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        eval_loss, eval_acc, eval_counter = eval_loss + \
            curr_eval_loss, eval_acc + curr_eval_acc, eval_counter + 1
    return eval_loss / float(eval_counter), eval_acc / float(eval_counter)

# 从logits中取出前五 get label using logits


def get_label_using_logits(logits, vocabulary_index2word_label, top_number=1):
    # print("get_label_using_logits.logits:",logits) #1-d array:
    # array([-5.69036102, -8.54903221, -5.63954401, ...,
    # -5.83969498,-5.84496021, -6.13911009], dtype=float32))
    index_list = np.argsort(logits)[-top_number:]
    index_list = index_list[::-1]
    # label_list=[]
    # for index in index_list:
    #    label=vocabulary_index2word_label[index]
    # label_list.append(label) #('get_label_using_logits.label_list:',
    # [u'-3423450385060590478', u'2838091149470021485',
    # u'-3174907002942471215', u'-1812694399780494968',
    # u'6815248286057533876'])
    return index_list

# 统计预测的准确率


def calculate_accuracy(labels_predicted, labels, eval_counter):
    label_nozero = []
    # print("labels:",labels)
    labels = list(labels)
    for index, label in enumerate(labels):
        if label > 0:
            label_nozero.append(index)
    if eval_counter < 2:
        print("labels_predicted:", labels_predicted,
              " ;labels_nozero:", label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)


if __name__ == "__main__":
    tf.app.run()
