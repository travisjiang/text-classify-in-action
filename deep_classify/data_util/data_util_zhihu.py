# -*- coding: utf-8 -*-
import codecs
import numpy as np
# load data of zhihu
import word2vec
import os
import pickle
from tflearn.data_utils import pad_sequences

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/..")

from config import zhihu_config


PAD_ID = 0
_GO = "_GO"
_END = "_END"
_PAD = "_PAD"


def create_vocabulary_from_model(word2vec_model_path, special_words):
    if not word2vec_model_path:
        return None,None
    vocab_word2index = {}
    vocab_index2word = {}
    print("create vocab. word2vec_model_path:", word2vec_model_path)

    model = word2vec.load(word2vec_model_path)

    counter = 0
    for word in special_words:
        vocab_word2index[word] = counter
        counter += 1

    for i, word in enumerate(model.vocab):
        vocab_word2index[word] = i + counter + special_index

    vocab_index2word = {i:w for w, i in vocab_word2index}

    return vocab_word2index, vocab_index2word


def create_vocabulary_from_data(data_set, special_words):
    vocab_word2index = {}
    vocab_index2word = {}
    counter = 0

    for word in special_words:
        vocab_word2index[word] = counter
        counter += 1

    for words in data_set:
        if isinstance(words, list):
            for word in words:
                if not vocab_word2index.get(word, None):
                    vocab_word2index[word] = counter
                    counter += 1
        else:
            if not vocab_word2index.get(words, None):
                vocab_word2index[words] = counter
                counter += 1

    vocab_index2word = {i:w for w, i in vocab_word2index}

    return vocab_word2index, vocab_index2word


def parse_train_file(path):
    with codecs.open(path, 'r', 'utf8') as f:

        line = f.readline()
        while line:
            x, y = line.split("__label__")
            y = y.strip().replace("\n", "").split(',')

            element_lists = x.split('\t')

            title_chars = [x for x in element_lists[0].strip().split(",")]
            title_words = [x for x in element_lists[1].strip().split(",")]
            desc_chars = [x for x in element_lists[2].strip().split(",")]
            desc_words = [x for x in element_lists[3].strip().split(",")]
            yield title_chars, title_words, desc_chars, desc_words, y

            line = f.readline()

def load_data(train_data_path,
        word2vec_model_path=zhihu_config['word_embedding'],
        char2vec_model_path=zhihu_config['char_embedding']):

    Y, X_title_char, X_title_word, X_desc_char, X_desc_word = []*5

    i = 0
    for tc, tw, dc, dw, y in parse_train_file(data_path):
        if i < 2:
            i += 1
            print(i, "x before:", [tc,tw,dc,dw])
            print(i, "y before:", y)

        X_title_char.append(tc)
        X_title_word.append(tw)
        X_desc_char.append(dc)
        X_desc_word.append(dw)
        Y.append(y)

    X = [X_title_char, X_title_word, X_desc_char, X_desc_words]

    return X, Y


def convert2index(word2index, X_words):
    X_indexs = []
    for row in X_words:
        X_indexs.append(word2index[w] for w in row)

    return X_indexs

def get_embedding(word2index, model_path):
    if not model_path:
        return None

    word2vec_model = word2vec.load(model_path)

    vocab_size = len(word2index)
    embed_size = len(word2vec_model.vectors[0])

    word_embedding = [None] * vocab_size

    # for words can be found in model
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        if word in word2index.keys():
            word_embedding[word2index[word]] = vector


    # for words cannot be found in model
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)

    for i in range(len(word2index)):
        if not word_embedding[i]:
            word_embedding[i] = np.random.uniform(-bound, bound, embed_size);

    # convert to 2d array
    word_embedding = np.array(word_embedding)

    return word_embedding


def create_voabulary_y(training_data_path=zhihu_config[train_set_question_topic],
                            name_scope='', use_seq2seq=False):
    print("create_voabulary_label_sorted.started.training_data_path:",
          training_data_path)
    zhihu_f_train = codecs.open(training_data_path, 'r', 'utf8')
    lines = zhihu_f_train.readlines()
    count = 0
    vocabulary_word2index_label = {}
    vocabulary_index2word_label = {}
    vocabulary_label_count_dict = {}  # {label:count}
    for i, line in enumerate(lines):
        if '__label__' in line:  # '__label__-2051131023989903826
            label = line[line.index('__label__') +
                         len('__label__'):].strip().replace("\n", "")
            if vocabulary_label_count_dict.get(label, None) is not None:
                vocabulary_label_count_dict[label] = vocabulary_label_count_dict[label] + 1
            else:
                vocabulary_label_count_dict[label] = 1
    list_label = sort_by_value(vocabulary_label_count_dict)

    # print(";list_label:",list_label)
    print("length of list_label:", len(list_label));
    countt = 0

    ##########################################################################
    for i, label in enumerate(list_label):
        if i < 10:
            count_value = vocabulary_label_count_dict[label]
            print("label:", label, "count_value:", count_value)
            countt = countt + count_value
        indexx = i + 3 if use_seq2seq else i
        vocabulary_word2index_label[label] = indexx
        vocabulary_index2word_label[indexx] = label
    print("count top10:", countt)

    print("create_voabulary_label_sorted.ended.len of vocabulary_label:",
          len(vocabulary_index2word_label))
    return vocabulary_word2index_label, vocabulary_index2word_label


def pad_sequences_str(trainX, maxlen=100, padding='post', truncating='post',
                      value=0.):
    pad_X = []
    for x in trainX:
        if len(trainX) > maxlen:
            if padding == 'post':
                pad_X.append(trainX[:maxlen])
            else:
                pad_X.append(trainX[-maxlen:])
        else:
            tmp_x = [value] * maxlen
            if padding == 'post':
                tmp_x[:len(x)] = x
            else:
                tmp_x[-len(x):] = x
            pad_X.append(tmp_x)
    return pad_X


class Vocabulary:
    def __init__(self, w2i=None, i2w=None, count=None):
        self.w2i = w2i
        self.i2w = i2w
        self.count = count

    def word2index(self, word, default_value):
        if self.w2i:
            if word in self.w2i.keys():
                return self.w2i[word]
            elif not self.i2w:
                self.count += 1
                self.w2i[word] = self.count
                return self.count
        else:
            self.w2i = {}
        return default_value

    def index2word(self, index, default_value):
        if not self.w2i:
            self.i2w = {i: w for w, i in self.w2i.items()}
        return self.i2w.get(index, default_value)

def parse_train_file(path):
    with codecs.open(path, 'r', 'utf8') as f:

        line = f.readline()
        while line:
            x, y = line.split("__label__")
            y = y.strip().replace("\n", "").split(',')

            element_lists = x.split('\t')

            title_chars = [x for x in element_lists[0].strip().split(",")]
            title_words = [x for x in element_lists[1].strip().split(",")]
            desc_chars = [x for x in element_lists[2].strip().split(",")]
            desc_words = [x for x in element_lists[3].strip().split(",")]
            yield title_chars, title_words, desc_chars, desc_words, y

            line = f.readline()

def load_data_none_embedding(train_data_path,
        paddings=zhihu_config['question_paddings_same'],
        label_path= zhihu_config['train_set_question_topic']):
    load_datalowmem(train_data_path, paddings, label_path, None, None)

def load_data_lowmem(train_data_path,
        use_static_embedding = False,
        paddings=zhihu_config['question_paddings_same'],
        label_path= zhihu_config['train_set_question_topic'],
        word2vec_model_path=zhihu_config['word_embedding'],
        char2vec_model_path=zhihu_config['char_embedding']):

    Y, X_title_char, X_title_word, X_desc_char, X_desc_word = []*5

    word2index, index2word = create_vocabulary_x(word2vec_model_path)
    char2index, index2char = create_vocabulary_x(char2vec_model_path)
    label2index, index2label = create_vocabulary_y(label_path)

    x_word_vocab = Vocabulary(word2index, index2word)
    x_char_vocab = Vocabulary(char2index, index2char)
    y_label_vocab = Vocabulary(label2index, index2label)

    i = 0
    for tc, tw, dc, dw, y in parse_train_file(data_path):
        if i < 2:
            i += 1
            print(i, "x before:", [tc,tw,dc,dw])
            print(i, "y before:", y)

        # if can't find the word, set the index as '0'.(equal to PAD_ID =
        # 0)
        tc = [x_char_vocab.word2index(c, 0) for c in tc]
        tw = [x_word_vacab.word2index(w, 0) for w in tw]
        dc = [x_char_vocab.word2index(c, 0) for c in dc]
        dw = [x_word_vocab.word2index(w, 0) for w in dw]
        y = [y_label_vocab.word2index(l, 0) for l in y]

        if i < 2:
            print(i, "x after:", [tc,tw,dc,dw])
            print(i, "y after:", y)

        X_title_char.append(tc)
        X_title_word.append(tw)
        X_desc_char.append(dc)
        X_desc_word.append(dw)
        Y.append(y)

    # pad sequences
    X = [X_title_char, X_title_word, X_desc_char, X_desc_word]
    assert len(X) == len(paddings)
    for i in range(len(X)):
       X[i] = pad_sequences(X[i], maxlen=paddings[i], value=0.)
    X_title_char, X_title_word, X_desc_char, X_desc_word = X

    #create embed
    x_char_embed = _create_embedding(x_char_vocab, char2vec_model_path)
    x_word_embed = _create_embedding(x_word_vocab, word2vec_model_path)
    E = [x_char_embed, x_word_embed, x_char_embed, x_word_embed]

    return X, Y, E

def get_embedding(word2index, model_path):
    if not model_path:
        return None

    word2vec_model = word2vec.load(model_path)

    vocab_size = len(word2index)
    embed_size = len(word2vec_model.vectors[0])

    word_embedding = [None] * vocab_size

    # for words can be found in model
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        if word in word2index.keys():
            word_embedding[word2index[word]] = vector


    # for words cannot be found in model
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)

    for i in range(len(word2index)):
        if not word_embedding[i]:
            word_embedding[i] = np.random.uniform(-bound, bound, embed_size);

    # convert to 2d array
    word_embedding = np.array(word_embedding)

    return word_embedding

def _create_embedding(vocab, model_path):
    if not model_path:
        return None

    word2vec_model = word2vec.load(model_path)
    vocab_size = len(word2vec_model.vocab)
    word_embedding = [None] * vocab_size  # create an empty word_embedding list.
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word_embedding[vocab.word2index[word]] = vector

    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    embed_size = len(word2vec_model.vectors[0])
    for i in range(vocab_size):
        if len(word_embedding[i]) != embed_size:
            word_embedding[i] = np.random.uniform(-bound, bound, embed_size);

    word_embedding = np.array(word_embedding)  # covert to 2d array.
    return word_embedding


def load_data_simple(vocabulary_word2index,
                     vocabulary_word2index_label,
                     valid_portion=0.05,
                     max_training_data=1000000,
                     training_data_path='train-zhihu4-only-title-all.txt',
                     multi_label_flag=False):  # n_words=100000,
    """
    input: a file path
    :return: train, test, valid. where train=(trainX, trainY). where
                trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    # 1.load a zhihu data from file
    # example:"w305 w6651 w3974 w1005 w54 w109 w110 w3974 w29 w25 w1513 w3645
    # w6 w111 __label__-400525901828896492"
    print("load_data.started...")
    print("load_data_multilabel_new.training_data_path:", training_data_path)
    # -zhihu4-only-title.txt
    zhihu_f = codecs.open(training_data_path, 'r', 'utf8')
    lines = zhihu_f.readlines()
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    Y_decoder_input = []  # ADD 2017-06-15
    for i, line in enumerate(lines):
        # x='w17314 w5521 w7729 w767 w10147 w111'
        x, y = line.split('__label__')
        y = y.strip().replace('\n', '')
        x = x.strip()
        if i < 1:
            print(i, "x0:", x)  # get raw x
        # x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        x = x.split(" ")
        # if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        x = [vocabulary_word2index.get(e, 0) for e in x]
        if i < 2:
            print(i, "x1:", x)  # word to index

        if multi_label_flag:  # 2)prepare multi-label format for classification
            ys = y.replace('\n', '').split(" ")  # ys is a list
            ys_index = []
            for y in ys:
                y_index = vocabulary_word2index_label[y]
                ys_index.append(y_index)
            ys_mulithot_list = transform_multilabel_as_multihot(ys_index)
        else:  # 3)prepare single label format for classification
            ys_mulithot_list = vocabulary_word2index_label[y]

        if i <= 3:
            print("ys_index:")
            # print(ys_index)
            # ," ;ys_decoder_input:",ys_decoder_input)
            print(i, "y:", y, " ;ys_mulithot_list:", ys_mulithot_list)
        X.append(x)
        Y.append(ys_mulithot_list)
        # if i>50000:
        #    break
    # 4.split to train,test and valid data
    number_examples = len(X)
    print("number_examples:", number_examples)
    train = (X[0:int((1 - valid_portion) * number_examples)],
             Y[0:int((1 - valid_portion) * number_examples)])
    test = (X[int((1 - valid_portion) * number_examples) + 1:],
            Y[int((1 - valid_portion) * number_examples) + 1:])
    # 5.return
    print("load_data.ended...")
    return train, test, test

# create vocabulary of lables. label is sorted. 1 is high frequency, 2 is
# low frequency.


def create_voabulary_label(voabulary_label='train-zhihu4-only-title-all.txt', name_scope='', use_seq2seq=False):  # 'train-zhihu.txt'
    print("create_voabulary_label_sorted.started.training_data_path:", voabulary_label)
    zhihu_f_train = codecs.open(voabulary_label, 'r', 'utf8')
    lines = zhihu_f_train.readlines()
    count = 0
    vocabulary_word2index_label = {}
    vocabulary_index2word_label = {}
    vocabulary_label_count_dict = {}  # {label:count}
    for i, line in enumerate(lines):
        if '__label__' in line:  # '__label__-2051131023989903826
            label = line[line.index('__label__') +
                         len('__label__'):].strip().replace("\n", "")
            if vocabulary_label_count_dict.get(label, None) is not None:
                vocabulary_label_count_dict[label] = vocabulary_label_count_dict[label] + 1
            else:
                vocabulary_label_count_dict[label] = 1
    list_label = sort_by_value(vocabulary_label_count_dict)

    # print(";list_label:",list_label)
    print("length of list_label:", len(list_label));
    countt = 0

    ##########################################################################
    for i, label in enumerate(list_label):
        if i < 10:
            count_value = vocabulary_label_count_dict[label]
            print("label:", label, "count_value:", count_value)
            countt = countt + count_value
        indexx = i + 3 if use_seq2seq else i
        vocabulary_word2index_label[label] = indexx
        vocabulary_index2word_label[indexx] = label
    print("count top10:", countt)

    print("create_voabulary_label_sorted.ended.len of vocabulary_label:",
          len(vocabulary_index2word_label))
    return vocabulary_word2index_label, vocabulary_index2word_label


def sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    return [backitems[i][1] for i in range(0, len(backitems))]


def create_voabulary_labelO():
    model = word2vec.load('zhihu-word2vec-multilabel.bin-100',
                          kind='bin')  # zhihu-word2vec.bin-100
    count = 0
    vocabulary_word2index_label = {}
    vocabulary_index2word_label = {}
    label_unique = {}
    for i, vocab in enumerate(model.vocab):
        if '__label__' in vocab:  # '__label__-2051131023989903826
            label = vocab[vocab.index('__label__') + len('__label__'):]
            if label_unique.get(label, None) is None:  # 不曾出现过的话，保持到字典中
                vocabulary_word2index_label[label] = count
                vocabulary_index2word_label[count] = label  # ADD
                count = count + 1
                label_unique[label] = label
    return vocabulary_word2index_label, vocabulary_index2word_label


def load_data_multilabel_new(vocabulary_word2index, vocabulary_word2index_label, valid_portion=0.05, max_training_data=1000000,
                             training_data_path='train-zhihu4-only-title-all.txt', multi_label_flag=True, use_seq2seq=False, seq2seq_label_length=6):  # n_words=100000,
    """
    input: a file path
    :return: train, test, valid. where train=(trainX, trainY). where
                trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    # 1.load a zhihu data from file
    # example:"w305 w6651 w3974 w1005 w54 w109 w110 w3974 w29 w25 w1513 w3645
    # w6 w111 __label__-400525901828896492"
    print("load_data.started...")
    print("load_data_multilabel_new.training_data_path:", training_data_path)
    # -zhihu4-only-title.txt
    zhihu_f = codecs.open(training_data_path, 'r', 'utf8')
    lines = zhihu_f.readlines()
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    Y_decoder_input = []  # ADD 2017-06-15
    for i, line in enumerate(lines):
        # x='w17314 w5521 w7729 w767 w10147 w111'
        x, y = line.split('__label__')
        y = y.strip().replace('\n', '')
        x = x.strip()
        if i < 1:
            print(i, "x0:", x)  # get raw x
        # x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        x = x.split(" ")
        # if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        x = [vocabulary_word2index.get(e, 0) for e in x]
        if i < 2:
            print(i, "x1:", x)  # word to index
        # 1)prepare label for seq2seq format(ADD _GO,_END,_PAD for seq2seq)
        if use_seq2seq:
            ys = y.replace('\n', '').split(" ")  # ys is a list
            _PAD_INDEX = vocabulary_word2index_label[_PAD]
            ys_mulithot_list = [_PAD_INDEX] * \
                seq2seq_label_length  # [3,2,11,14,1]
            ys_decoder_input = [_PAD_INDEX] * seq2seq_label_length
            # below is label.
            for j, y in enumerate(ys):
                if j < seq2seq_label_length - 1:
                    ys_mulithot_list[j] = vocabulary_word2index_label[y]
            if len(ys) > seq2seq_label_length - 1:
                # ADD END TOKEN
                ys_mulithot_list[seq2seq_label_length -
                                 1] = vocabulary_word2index_label[_END]
            else:
                ys_mulithot_list[len(ys)] = vocabulary_word2index_label[_END]

            # below is input for decoder.
            ys_decoder_input[0] = vocabulary_word2index_label[_GO]
            for j, y in enumerate(ys):
                if j < seq2seq_label_length - 1:
                    ys_decoder_input[j + 1] = vocabulary_word2index_label[y]
            if i < 10:
                print(i, "ys:==========>0", ys)
                print(i, "ys_mulithot_list:==============>1", ys_mulithot_list)
                print(i, "ys_decoder_input:==============>2", ys_decoder_input)
        else:
            if multi_label_flag:  # 2)prepare multi-label format for classification
                ys = y.replace('\n', '').split(" ")  # ys is a list
                ys_index = []
                for y in ys:
                    y_index = vocabulary_word2index_label[y]
                    ys_index.append(y_index)
                ys_mulithot_list = transform_multilabel_as_multihot(ys_index)
            else:  # 3)prepare single label format for classification
                ys_mulithot_list = vocabulary_word2index_label[y]
        if i <= 3:
            print("ys_index:")
            # print(ys_index)
            # ," ;ys_decoder_input:",ys_decoder_input)
            print(i, "y:", y, " ;ys_mulithot_list:", ys_mulithot_list)
        X.append(x)
        Y.append(ys_mulithot_list)
        if use_seq2seq:
            Y_decoder_input.append(ys_decoder_input)  # decoder input
        # if i>50000:
        #    break
    # 4.split to train,test and valid data
    number_examples = len(X)
    print("number_examples:", number_examples)
    train = (X[0:int((1 - valid_portion) * number_examples)],
             Y[0:int((1 - valid_portion) * number_examples)])
    test = (X[int((1 - valid_portion) * number_examples) + 1:],
            Y[int((1 - valid_portion) * number_examples) + 1:])
    if use_seq2seq:
        train = train + \
            (Y_decoder_input[0:int((1 - valid_portion) * number_examples)],)
        test = test + \
            (Y_decoder_input[int((1 - valid_portion) * number_examples) + 1:],)
    # 5.return
    print("load_data.ended...")
    return train, test, test


def load_data_multilabel_new_twoCNN(vocabulary_word2index, vocabulary_word2index_label, valid_portion=0.05, max_training_data=1000000,
                                    training_data_path='train-zhihu4-only-title-all.txt', multi_label_flag=True):  # n_words=100000,
    """
    input: a file path
    :return: train, test, valid. where train=(trainX, trainY). where
                trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    # 1.load a zhihu data from file
    # example:"w305 w6651 w3974 w1005 w54 w109 w110 w3974 w29 w25 w1513 w3645
    # w6 w111 __label__-400525901828896492"
    print("load_data.twoCNN.started...")
    print("load_data_multilabel_new_twoCNN.training_data_path:", training_data_path)
    # -zhihu4-only-title.txt
    zhihu_f = codecs.open(training_data_path, 'r', 'utf8')
    lines = zhihu_f.readlines()
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    X2 = []
    Y = []
    count_error = 0
    for i, line in enumerate(lines):
        # x='w17314 w5521 w7729 w767 w10147 w111'
        x, y = line.split('__label__')
        y = y.strip().replace('\n', '')
        x = x.strip()
        # print("x:===============>",x)
        try:
            x, x2 = x.split("\t")
        except Exception:
            print("x.split.error.", x, "count_error:", count_error)
            count_error += 1
            continue
        if i < 1:
            print(i, "x0:", x)  # get raw x
        # x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        x = x.split(" ")
        # if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        x = [vocabulary_word2index.get(e, 0) for e in x]
        x2 = x2.split(" ")
        x2 = [vocabulary_word2index.get(e, 0) for e in x2]
        if i < 1:
            print(i, "x1:", x, "x2:", x2)  # word to index
        if multi_label_flag:
            ys = y.replace('\n', '').split(" ")  # ys is a list
            ys_index = []
            for y in ys:
                y_index = vocabulary_word2index_label[y]
                ys_index.append(y_index)
            ys_mulithot_list = transform_multilabel_as_multihot(ys_index)
        else:
            ys_mulithot_list = int(y)  # vocabulary_word2index_label[y]
        if i < 1:
            print(i, "y:", y, "ys_mulithot_list:", ys_mulithot_list)
        X.append(x)
        X2.append(x2)
        Y.append(ys_mulithot_list)
    # 4.split to train,test and valid data
    number_examples = len(X)
    print("number_examples:", number_examples)
    train = (X[0:int((1 - valid_portion) * number_examples)], X2[0:int((1 - valid_portion)
                                                                       * number_examples)], Y[0:int((1 - valid_portion) * number_examples)])
    test = (X[int((1 - valid_portion) * number_examples) + 1:], X2[int((1 - valid_portion)
                                                                       * number_examples) + 1:], Y[int((1 - valid_portion) * number_examples) + 1:])
    # 5.return
    print("load_data.ended...")
    return train, test, test


def load_data(vocabulary_word2index, vocabulary_word2index_label, valid_portion=0.05, max_training_data=1000000, training_data_path='train-zhihu4-only-title-all.txt'):  # n_words=100000,
    """
    input: a file path
    :return: train, test, valid. where train=(trainX, trainY). where
                trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    # 1.load a zhihu data from file
    # example:"w305 w6651 w3974 w1005 w54 w109 w110 w3974 w29 w25 w1513 w3645
    # w6 w111 __label__-400525901828896492"
    print("load_data.started...")
    zhihu_f = codecs.open(training_data_path, 'r',
                          'utf8')  # -zhihu4-only-title.txt
    lines = zhihu_f.readlines()
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    for i, line in enumerate(lines):
        # x='w17314 w5521 w7729 w767 w10147 w111'
        x, y = line.split('__label__')
        y = y.replace('\n', '')
        x = x.replace("\t", ' EOS ').strip()
        if i < 5:
            print("x0:", x)  # get raw x
        # x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        # if i<5:
        #    print("x1:",x_) #
        x = x.split(" ")
        # if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        x = [vocabulary_word2index.get(e, 0) for e in x]
        if i < 5:
            print("x1:", x)  # word to index
        y = vocabulary_word2index_label[y]  # np.abs(hash(y))
        X.append(x)
        Y.append(y)
    # 4.split to train,test and valid data
    number_examples = len(X)
    print("number_examples:", number_examples)
    train = (X[0:int((1 - valid_portion) * number_examples)],
             Y[0:int((1 - valid_portion) * number_examples)])
    test = (X[int((1 - valid_portion) * number_examples) + 1:],
            Y[int((1 - valid_portion) * number_examples) + 1:])
    # 5.return
    print("load_data.ended...")
    return train, test, test

 # 将一句话转化为(uigram,bigram,trigram)后的字符串


def process_one_sentence_to_get_ui_bi_tri_gram(sentence, n_gram=3):
    """
    :param sentence: string. example:'w17314 w5521 w7729 w767 w10147 w111'
    :param n_gram:
    :return:string. example:'w17314 w17314w5521 w17314w5521w7729 w5521 w5521w7729 w5521w7729w767 w7729 w7729w767 w7729w767w10147 w767 w767w10147 w767w10147w111 w10147 w10147w111 w111'
    """
    result = []
    # [sentence[i] for i in range(len(sentence))]
    word_list = sentence.split(" ")
    unigram = ''
    bigram = ''
    trigram = ''
    fourgram = ''
    length_sentence = len(word_list)
    for i, word in enumerate(word_list):
        unigram = word  # ui-gram
        word_i = unigram
        if n_gram >= 2 and i + 2 <= length_sentence:  # bi-gram
            bigram = "".join(word_list[i:i + 2])
            word_i = word_i + ' ' + bigram
        if n_gram >= 3 and i + 3 <= length_sentence:  # tri-gram
            trigram = "".join(word_list[i:i + 3])
            word_i = word_i + ' ' + trigram
        if n_gram >= 4 and i + 4 <= length_sentence:  # four-gram
            fourgram = "".join(word_list[i:i + 4])
            word_i = word_i + ' ' + fourgram
        if n_gram >= 5 and i + 5 <= length_sentence:  # five-gram
            fivegram = "".join(word_list[i:i + 5])
            word_i = word_i + ' ' + fivegram
        result.append(word_i)
    result = " ".join(result)
    return result

# 加载数据，标签包含多个label：load data with multi-labels


def load_data_with_multilabels(vocabulary_word2index, vocabulary_word2index_label, traning_path, valid_portion=0.05, max_training_data=1000000):  # n_words=100000,
    """
    input: a file path
    :return: train, test, valid. where train=(trainX, trainY). where
                trainX: is a list of list.each list representation a sentence.trainY: is a list of label. each label is a number
    """
    # 1.load a zhihu data from file
    # example: 'w140 w13867 w10344 w2673 w9514 w269 w460 w6 w35053 w844 w10147
    # w111 __label__-2379261820462209275 -5535923551616745326
    # 6038661761506862294'
    print("load_data_with_multilabels.ended...")
    # ('/home/xul/xul/9_ZhihuCup/'+data_type+'-zhihu5-only-title-multilabel.txt', 'r', 'utf8') #home/xul/xul/9_ZhihuCup/'
    zhihu_f = codecs.open(traning_path, 'r', 'utf8')
    lines = zhihu_f.readlines()
    # 2.transform X as indices
    # 3.transform  y as scalar
    X = []
    Y = []
    Y_label1999 = []
    for i, line in enumerate(lines):
        # if i>max_training_data:
        #    break
        # x='w17314 w5521 w7729 w767 w10147 w111'
        x, ys = line.split('__label__')
        ys = ys.replace('\n', '').split(" ")
        x = x.strip()
        if i < 5:
            print("x0:", x)  # u'w4260 w4260w86860 w4260w86860w30907 w86860 w86860w30907 w86860w30907w11 w30907 w30907w11 w30907w11w31 w11 w11w31 w11w31w72 w31 w31w72 w31w72w166 w72 w72w166 w72w166w346 w166 w166w346 w166w346w2182 w346 w346w2182 w346w2182w224 w2182 w2182w224 w2182w224w2148 w224 w224w2148 w224w2148w6 w2148 w2148w6 w2148w6w2566 w6 w6w2566 w6w2566w25 w2566 w2566w25 w2566w25w1110 w25 w25w1110 w25w1110w111 w1110 w1110w111 w111'
        # x_=process_one_sentence_to_get_ui_bi_tri_gram(x)
        # if i < 5:
        #    print("x1:", x_)
        x = x.split(" ")
        # if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        x = [vocabulary_word2index.get(e, 0) for e in x]
        if i < 5:
            print("x2:", x)
        # print("ys:",ys) #['501174938575526146', '-4317515119936650885']
        ys_list = []
        for y in ys:
            y_ = vocabulary_word2index_label[y]
            ys_list.append(y_)
        X.append(x)
        # TODO ys_list_array=transform_multilabel_as_multihot(ys_list) #it is
        # 2-d array. [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
        # 0.] [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]...]
        ys_list_ = proces_label_to_algin(ys_list)
        Y.append(ys_list_)
        # TODO Y_label1999.append(ys_list_array)
        if i == 0:
            print(X, Y)
            print(Y_label1999)
    # 4.split to train,test and valid data
    number_examples = len(X)
    # TODO Y_label1999[0:int((1 - valid_portion) * number_examples)]
    train = (X[0:int((1 - valid_portion) * number_examples)],
             Y[0:int((1 - valid_portion) * number_examples)])
    # TODO ,Y_label1999[int((1 - valid_portion) * number_examples) + 1:]
    test = (X[int((1 - valid_portion) * number_examples) + 1:],
            Y[int((1 - valid_portion) * number_examples) + 1:])
    print("load_data_with_multilabels.ended...")
    return train, test

# 将LABEL转化为MULTI-HOT


# 1999label_list=[0,1,4,9,5]
def transform_multilabel_as_multihot(label_list, label_size=1999):
    """
    :param label_list: e.g.[0,1,4]
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result = np.zeros(label_size)
    # set those location as 1, all else place as 0.
    result[label_list] = 1
    return result

# 将LABEL转化为MULTI-HOT


# 1999label_list=[0,1,4,9,5]
def transform_multilabel_as_multihotO(label_list, label_size=1999):
    batch_size = len(label_list)
    result = np.zeros((batch_size, label_size))
    # set those location as 1, all else place as 0.
    result[(range(batch_size), label_list)] = 1
    return result


def load_final_test_data(file_path):
    final_test_file_predict_object = codecs.open(file_path, 'r', 'utf8')
    lines = final_test_file_predict_object.readlines()
    question_lists_result = []
    for i, line in enumerate(lines):
        question_id, question_string = line.split("\t")
        question_string = question_string.strip().replace("\n", "")
        question_lists_result.append((question_id, question_string))
    print("length of total question lists:", len(question_lists_result))
    return question_lists_result


def load_data_predict(vocabulary_word2index, vocabulary_word2index_label, questionid_question_lists, uni_to_tri_gram=False):  # n_words=100000,
    final_list = []
    for i, tuplee in enumerate(questionid_question_lists):
        queston_id, question_string_list = tuplee
        if uni_to_tri_gram:
            x_ = process_one_sentence_to_get_ui_bi_tri_gram(
                question_string_list)
            x = x_.split(" ")
        else:
            x = question_string_list.split(" ")
        # if can't find the word, set the index as '0'.(equal to PAD_ID = 0)
        x = [vocabulary_word2index.get(e, 0) for e in x]
        if i <= 2:
            print("question_id:", queston_id); print(
                "question_string_list:", question_string_list); print("x_indexed:", x)
        final_list.append((queston_id, x))
    number_examples = len(final_list)
    print("number_examples:", number_examples)
    return final_list


def proces_label_to_algin(ys_list, require_size=5):
    """
    :param ys_list: a list
    :return: a list
    """
    ys_list_result = [0 for x in range(require_size)]
    if len(ys_list) >= require_size:  # 超长
        ys_list_result = ys_list[0:require_size]
    else:  # 太短
        if len(ys_list) == 1:
            ys_list_result = [ys_list[0] for x in range(require_size)]
        elif len(ys_list) == 2:
            ys_list_result = [ys_list[0], ys_list[0],
                              ys_list[0], ys_list[1], ys_list[1]]
        elif len(ys_list) == 3:
            ys_list_result = [ys_list[0], ys_list[0],
                              ys_list[1], ys_list[1], ys_list[2]]
        elif len(ys_list) == 4:
            ys_list_result = [ys_list[0], ys_list[0],
                              ys_list[1], ys_list[2], ys_list[3]]
    return ys_list_result


def write_uigram_to_trigram():
    pass
    # 1.read file.
    # 2.uigram--->trigram
    # 3.write each line to file system.


def test_pad():
    trainX = 'w18476 w4454 w1674 w6 w25 w474 w1333 w1467 w863 w6 w4430 w11 w813 w4463 w863 w6 w4430 w111'
    trainX = trainX.split(" ")
    trainX = pad_sequences([trainX], maxlen=100, value=0.)
    print("trainX:", trainX)


topic_info_file_path = 'topic_info.txt'


def read_topic_info():
    f = codecs.open(topic_info_file_path, 'r', 'utf8')
    lines = f.readlines()
    dict_questionid_title = {}
    for i, line in enumerate(lines):
        topic_id, partent_ids, title_character, title_words, desc_character, decs_words = line.split(
            "\t").strip()
        # print(i,"------------------------------------------------------")
        # print("topic_id:",topic_id)
        # print("partent_ids:",partent_ids)
        # print("title_character:",title_character)
        # print("title_words:",title_words)
        # print("desc_character:",desc_character)
        # print("decs_words:",decs_words)
        dict_questionid_title[topic_id] = title_words + " " + decs_words
    print("len(dict_questionid_title):", len(dict_questionid_title))
    return dict_questionid_title


def stat_training_data_length():
    training_data = 'train-zhihu4-only-title-all.txt'
    f = codecs.open(training_data, 'r', 'utf8')
    lines = f.readlines()
    length_dict = {0: 0, 5: 0, 10: 0, 15: 0, 20: 0, 25: 0,
                   30: 0, 35: 0, 40: 0, 100: 0, 150: 0, 200: 0, 1500: 0}
    length_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 100, 150, 200, 1500]
    for i, line in enumerate(lines):
        line_list = line.split('__label__')[0].strip().split(" ")
        length = len(line_list)
        # print(i,"length:",length)
        for l in length_list:
            if length < l:
                length = l
                # print("length.assigned:",length)
                break
        #print("length.before dict assign:", length)
        length_dict[length] = length_dict[length] + 1
    print("length_dict:", length_dict)


if __name__ == '__main__':
    if __name__ == '__main__':
        if __name__ == '__main__':
            # 1.
            #vocabulary_word2index, vocabulary_index2word=create_voabulary()
            #vocabulary_word2index_label, vocabulary_index2word_label=create_voabulary_label()
            # load_data_with_multilabels(vocabulary_word2index,vocabulary_word2index_label,data_type='test')
            # 2.
            # sentence=u'我想开通创业板'
            #sentence='w18476 w4454 w1674 w6 w25 w474 w1333 w1467 w863 w6 w4430 w11 w813 w4463 w863 w6 w4430 w111'
            # result=process_one_sentence_to_get_ui_bi_tri_gram(sentence,n_gram=3)
            # print(len(result),"result:",result)

            # 3. transform to multilabel
            # label_list=[0,1,4,9,5]
            # result=transform_multilabel_as_multihot(label_list,label_size=15)
            # print("result:",result)

            # 4.load data for predict------------------------------------------
            # file_path='test-zhihu-forpredict-v4only-title.txt'
            # questionid_question_lists=load_final_test_data(file_path)

            #vocabulary_word2index, vocabulary_index2word=create_voabulary()
            # vocabulary_word2index_label,_=create_voabulary_label()
            #final_list=load_data_predict(vocabulary_word2index, vocabulary_word2index_label, questionid_question_lists)

            # 5.process label require lengh
            # ys_list=[99999]
            # ys_list_result=proces_label_to_algin(ys_list,require_size=5)
            # print(ys_list,"ys_list_result1.:",ys_list_result)
            # ys_list=[99999,23423432,67566765]
            # ys_list_result=proces_label_to_algin(ys_list,require_size=5)
            # print(ys_list,"ys_list_result2.:",ys_list_result)
            # ys_list=[99999,23423432,67566765,23333333]
            # ys_list_result=proces_label_to_algin(ys_list,require_size=5)
            # print(ys_list,"ys_list_result2.:",ys_list_result)
            #ys_list = [99999, 23423432, 67566765,44543543,546546546,323423434]
            #ys_list_result = proces_label_to_algin(ys_list, require_size=5)
            #print(ys_list, "ys_list_result3.:", ys_list_result)

            # 6.create vocabulary label. sorted.
            # create_voabulary_label()

            # d={'a':3,'b':2,'c':11}
            # d_=sort_by_value(d)
            # print("d_",d_)

            # 7.
            # test_pad()

            # 8.read topic info
            # read_topic_info()

            # 9。
            stat_training_data_length()
