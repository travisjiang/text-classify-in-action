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

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + "/..")

from config import zhihu_config


q_t = zhihu_config["question_topic_train_set"]
q = zhihu_config["question_train_set"]


small_data_set_size = zhihu_config['small_data_set_size']
small_train_set_title = zhihu_config['train_set_small_only_title']
small_test_set_title = zhihu_config['test_set_small_only_title']
small_valid_set_title = zhihu_config['valid_set_small_only_title']

mid_data_set_size = zhihu_config['mid_data_set_size']
mid_train_set_title = zhihu_config['train_set_mid_only_title']
mid_test_set_title = zhihu_config['test_set_mid_only_title']
mid_valid_set_title = zhihu_config['valid_set_mid_only_title']

full_train_set_title = zhihu_config['train_set_full_only_title']
full_test_set_title = zhihu_config['test_set_full_only_title']
full_valid_set_title = zhihu_config['valid_set_full_only_title']

tmp_train_set_title = 'zhihu_train_data_tmp.txt'
tmp_test_set_title = 'zhihu_test_data_tmp.txt'
tmp_valid_set_title = 'zhihu_valid_data_tmp.txt'

using_top1_model = True  # only keeps the top1 topic for each question
multi_label_flag = False
cut_data_flag = True
output_data_size = 10000

# just for test
train_zhihu = tmp_train_set_title
test_zhihu = tmp_test_set_title
valid_zhihu = tmp_valid_set_title

output_flag = 2
if output_flag == 1:
    output_data_size = small_data_set_size
    train_zhihu = small_train_set_title
    test_zhihu = small_test_set_title
    valid_zhihu = small_test_set_title
elif output_flag == 2:
    output_data_size = mid_data_set_size
    train_zhihu = mid_train_set_title
    test_zhihu = mid_test_set_title
    valid_zhihu = mid_test_set_title
elif output_flag == 3:
    cut_data_flag = False
    train_zhihu = full_train_set_title
    test_zhihu = full_test_set_title
    valid_zhihu = full_test_set_title


data_size_threshold = output_data_size * 5 if cut_data_flag else 3000000

#1.#######################################################################
print("process question_topic_train_set.txt,started...")
q_t_file = codecs.open(q_t, 'r', 'utf8')
question_topic_dict = {}

line = q_t_file.readline()
i = 0
while line:
    i += 1
    if i % 300000 == 0:
        print(i)
    # print(line)
    question_id, topic_list_string = line.split('\t')
    # print(question_id)
    # print(topic_list_string)
    topic_list = topic_list_string.replace("\n", "").split(",")
    question_topic_dict[question_id] = topic_list
    line = q_t_file.readline()
    # for ii,topic in enumerate(topic_list):
    #    print(ii,topic)
    # print("=====================================")
    # if i>10:
    #   print(question_topic_dict)
    #   break
    if i > data_size_threshold:
        break
print("process question_topic_train_set.txt,ended...")
##########################################################################
##########################################################################
# 2.处理问题--得到问题ID：问题的表示，存成字典。proces question. for every question form a a
# list of string to reprensent it.
import codecs
print("process question started11...")
q_file = codecs.open(q, 'r', 'utf8')
questionid_words_representation = {}
question_representation = []
length_desc = 30

line = q_file.readline()
i = 0
while line:
    i += 1
    if i % 300000 == 0:
        print(i)
    # print("line:")
    # print(line)
    element_lists = line.split('\t')  # ['c324,c39','w305...','c']
    question_id = element_lists[0]
    # print("question_id:",element_lists[0])
    # for i,q_e in enumerate(element_lists):
    #    print("e:",q_e)
    # question_representation=[x for x in element_lists[2].split(",")] #+
    # #TODO this is only for title's word. no more.
    title_words = [x for x in element_lists[2].strip().split(",")
                   ][-length_desc:]
    # print("title_words:",title_words)
    title_c = [x for x in element_lists[1].strip().split(",")][-length_desc:]
    #print("title_c:", title_c)
    desc_words = [x for x in element_lists[4].strip().split(",")
                  ][-length_desc:]
    #print("desc_words:", desc_words)
    desc_c = [x for x in element_lists[3].strip().split(",")][-length_desc:]
    #print("desc_c:", desc_c)
    question_representation = title_words + title_c + desc_words + desc_c

    # questionid_words_representation[question_id]=question_representation
    ###############################################################
    # To simplify model, use just title words
    ###############################################################
    questionid_words_representation[question_id] = title_words
    line = q_file.readline()
    if i > data_size_threshold:
        break
q_file.close()
print("proces question ended2...")
##########################################################################
##########################################################################
# 3.获得模型需要的训练数据。以{问题的表示：TOPIC_ID}的形式的列表
# save training data,testing data: question __label__topic_id
import codecs
import random

print("saving traininig data.started1...")
count = 0
data_list = []
#topic_set = set()


def split_list(listt):
    random.shuffle(listt)
    list_len = len(listt)
    train_len = 0.95
    valid_len = 0.025
    train = listt[0:int(list_len * train_len)]
    valid = listt[int(list_len * train_len)                  :int(list_len * (train_len + valid_len))]
    test = listt[int(list_len * (train_len + valid_len)):]
    return train, valid, test


for question_id, question_representation in questionid_words_representation.items():
    # print("===================>")
    # print('question_id',question_id)
    # print("question_representation:",question_representation)
    # get label_id for this question_id by using:question_topic_dict
    topic_list = question_topic_dict[question_id]
    #topic_set = topic_set.union(set(topic_list))
    # print("topic_list:",topic_list)
    # if count>5:
    #    ii=0
    #    ii/0
    if not multi_label_flag:
        if using_top1_model:
            if len(topic_list) > 0:
                # single-label
                data_list.append((question_representation, topic_list[0]))
        else:
            for topic_id in topic_list:
                # single-label
                data_list.append((question_representation, topic_id))
    else:
        data_list.append((question_representation, topic_list))  # multi-label
    count = count + 1

# random shuffle list
random.shuffle(data_list)

##########################################################################
##########################################################################
# 4. 根据需要裁剪数据模型获得较小模型。以{问题的表示：TOPIC_ID}的形式的列表
# save training data,testing data: question __label__topic_id
print("saving traininig data.started2...")
small_count = 0
small_topic_size = 0

#import pdb
# pdb.set_trace()

if not multi_label_flag and cut_data_flag:
    print("cut traininig data to small size from orig_size %d..." % len(data_list))
    topic_question_dict = {}
    for q, t in data_list:
        if t not in topic_question_dict.keys():
            topic_question_dict[t] = []
        topic_question_dict[t].append(q)
    topic_question_dict = sorted(topic_question_dict.items(),
                                 key=lambda d: len(d[1]), reverse=True)

    small_data_list = []
    for t, q_list in topic_question_dict:
        small_data_list.extend([(q, t) for q in q_list])
        small_topic_size += 1

        print("topic %s, contains question %d, now data_size %d" % (t,
                                                                    len(q_list),
                                                                    len(small_data_list)))
        if len(small_data_list) >= output_data_size:
            break
    data_list = small_data_list
    # random shuffle list
    random.shuffle(data_list)
    print("cut traininig data to small size, new_size %d, topic_size %d..." %
          (len(data_list), small_topic_size))

##########################################################################
##########################################################################
# 5. 存储结果到本地文件系统。以{问题的表示：TOPIC_ID}的形式的列表


def write_data_to_file_system(file_name, data):
    file = codecs.open(file_name, 'a', 'utf8')
    for d in data:
        # print(d)
        question_representation, topic_id = d
        question_representation_ = " ".join(question_representation)
        file.write(question_representation_ +
                   " __label__" + str(topic_id) + "\n")
    file.close()


def write_data_to_file_system_multilabel(file_name, data):
    file = codecs.open(file_name, 'a', 'utf8')
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
