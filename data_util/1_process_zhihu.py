# -*- coding: utf-8 -*-
import sys
import os
# reload(sys)
# sys.setdefaultencoding('utf8')
# 1.将问题ID和TOPIC对应关系保持到字典里：process question_topic_train_set.txt
# from:question_id,topics(topic_id1,topic_id2,topic_id3,topic_id4,topic_id5)
#  to:(question_id,topic_id1)
#     (question_id,topic_id2)
# read question_topic_train_set.txt
import codecs
import random

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/..")

from config import zhihu_config

# input files
q_t = zhihu_config["question_topic_train_set"]
q = zhihu_config["question_train_set"]

# output files
cutted_data_size = zhihu_config['cutted_data_size']
data_size_threshold = cutted_data_size * 5

train_zhihu = zhihu_config['train_set_question_topic']
test_zhihu = zhihu_config['test_set_question_topic']
valid_zhihu = zhihu_config['valid_set_question_topic']

multi_label_flag = False


#1.#######################################################################
print("process question_topic_train_set.txt,started...")
questionid_topic_dict = {}

with codecs.open(q_t, 'r', 'utf8') as q_t_file:
    line = q_t_file.readline()
    i = 0
    while line:
        i += 1
        if i % 300000 == 0:
            print(i)
        question_id, topic_list_string = line.split('\t')
        topic_list = topic_list_string.replace("\n", "").split(",")
        questionid_topic_dict[question_id] = topic_list

        line = q_t_file.readline()

        if i > data_size_threshold:
            break
print("process question_topic_train_set.txt,ended...")
##########################################################################
##########################################################################
# 2.proces question. get dict={question:topic}
import codecs
print("process question_train_set.txt,started...")
question_representation_topic_dict = {}
topic_2_question_representation = {}
length_desc = 30

with codecs.open(q, 'r', 'utf8') as q_file:
    line = q_file.readline()
    i = 0
    while line:
        i += 1
        if i % 300000 == 0:
            print(i)
        element_lists = line.split('\t')  # ['c324,c39','w305...','c']
        question_id = element_lists[0]

        #title_chars = [x for x in element_lists[1].strip().split(",")
        #               ][-length_desc:]
        #title_words = [x for x in element_lists[2].strip().split(",")
        #               ][-length_desc:]
        #desc_chars = [x for x in element_lists[3].strip().split(",")
        #              ][-length_desc:]
        #desc_words = [x for x in element_lists[4].strip().split(",")
        #              ][-length_desc:]

        #question_representation = []
        #question_representation = title_words + title_chars + desc_words + desc_chars
        #question_representation_string = "\t".join(question_representation)
        question_representation_string = "\t".join(element_lists[1:])

        # question_representation_topic_dict[question_representation_string] = \
        #                                        questionid_topic_dict[question_id]

        topic_id = questionid_topic_dict[question_id][0] #only top1 topic
        if topic_id not in topic_2_question_representation.keys():
            topic_2_question_representation[topic_id] = []
        topic_2_question_representation[topic_id].append(question_representation_string)

        line = q_file.readline()
        if i > data_size_threshold:
            break
del questionid_topic_dict  # free memory

print("process question_train_set.txt,ended...")

##########################################################################
##########################################################################
# 3. 根据需要裁剪数据模型获得较小模型。以{问题的表示：TOPIC_ID}的形式的列表
# save training data,testing data: question __label__topic_id
print("cut traininig data.started2...")
cutted_count = 0
cutted_topic_size = 0

#import pdb
# pdb.set_trace()

print("cut traininig data to small size %d..." % cutted_data_size)
topic_2_question_representation = sorted(
    topic_2_question_representation.items(), key=lambda d: len(d[1]), reverse=True)

cutted_data_list = []
for t, q_list in topic_2_question_representation:
    cutted_data_list.extend([(q, t) for q in q_list])
    cutted_topic_size += 1

    print("topic %s, contains question %d, now data_size %d" % (t,
                                                                len(q_list),
                                                                len(cutted_data_list)))
    if len(cutted_data_list) >= cutted_data_size:
        break
data_list = cutted_data_list

del topic_2_question_representation

# random shuffle list
random.shuffle(data_list)
print("cut traininig data to small size, new_size %d, topic_size %d..." %
      (len(data_list), cutted_topic_size))

##########################################################################
##########################################################################
# 5. 存储结果到本地文件系统。以{问题的表示：TOPIC_ID}的形式的列表


def split_list(listt):
    random.shuffle(listt)
    list_len = len(listt)
    train_len = 0.95
    valid_len = 0.025
    train = listt[0:int(list_len * train_len)]
    valid = listt[int(list_len * train_len)
                      :int(list_len * (train_len + valid_len))]
    test = listt[int(list_len * (train_len + valid_len)):]
    return train, valid, test


def write_data_to_file_system(file_name, data):
    file = codecs.open(file_name, 'w', 'utf8')
    for d in data:
        # print(d)
        question_representation, topic_id = d
        file.write(question_representation +
                   " __label__" + str(topic_id) + "\n")
    file.close()


def write_data_to_file_system_multilabel(file_name, data):
    file = codecs.open(file_name, 'w', 'utf8')
    for d in data:
        question_representation, topic_id_list = d
        topic_id_list_ = " ".join(topic_id_list)
        file.write(question_representation + " __label__" +
                   str(topic_id_list_) + "\n")
    file.close()


train_data, valid_data, test_data = split_list(data_list)
if not multi_label_flag:  # single label
    write_data_to_file_system(train_zhihu, train_data)
    write_data_to_file_system(valid_zhihu, valid_data)
    write_data_to_file_system(test_zhihu, test_data)
else:  # multi-label
    write_data_to_file_system_multilabel(train_zhihu, train_data)
    write_data_to_file_system_multilabel(valid_zhihu, valid_data)
    write_data_to_file_system_multilabel(test_zhihu, test_data)

print("saving traininig data.ended...")
##########################################################################
