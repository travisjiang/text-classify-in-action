#!/usr/bin/env python
# coding=utf-8

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

zhihu_data_set_path = dir_path + '/data_set/ieee_zhihu_cup/'

cutted_data_size = 60000

preprocessed_data_set_path = dir_path + \
    '/data_set/preprocessed_data/' + str(cutted_data_size) + '_'

checkpoint_dir = dir_path + 'data_set/checkpoint'

vocabulary_cache = dir_path + 'data_set/cache_vocabulary_label_pik'

zhihu_config = {
    # path of data_set from zhihu
    'question_topic_train_set': zhihu_data_set_path + 'question_topic_train_set.txt',
    'question_train_set': zhihu_data_set_path + 'question_train_set.txt',
    'question_eval_set': zhihu_data_set_path + 'question_eval_set.txt',
    'topic_info': zhihu_data_set_path + 'topic_info.txt',
    'char_embedding': zhihu_data_set_path + 'char_embedding.txt',
    'word_embedding': zhihu_data_set_path + 'word_embedding.txt',

    # path of preprocessing data_set
    'cutted_data_size': cutted_data_size,
    'train_set_question_topic': preprocessed_data_set_path + 'train_set_question_topic.txt',
    'test_set_question_topic': preprocessed_data_set_path + 'test_set_question_topic.txt',
    'valid_set_question_topic': preprocessed_data_set_path + 'valid_set_question_topic.txt',

    # checkpoint
    'checkpoint_dir': checkpoint_dir,

    # cache
    'vocabulary_cache': vocabulary_cache

}
