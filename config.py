#!/usr/bin/env python
# coding=utf-8

import os

dir_path = os.path.dirname(os.path.realpath(__file__))

zhihu_data_set_path = dir_path + '/data_set/ieee_zhihu_cup/'

preprocessed_data_set_path = dir_path + '/data_set/preprocessed_data/'

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
    'small_data_set_size' : 60000,
    'train_set_small_only_title': preprocessed_data_set_path + 'train_set_small_only_title.txt',
    'test_set_small_only_title': preprocessed_data_set_path + 'test_set_small_only_title.txt',
    'valid_set_small_only_title': preprocessed_data_set_path + 'valid_set_small_only_title.txt',

    'train_set_small_title_desc': preprocessed_data_set_path + 'train_set_small_title_desc.txt',
    'test_set_small_title_desc': preprocessed_data_set_path + 'test_set_small_title_desc.txt',
    'valid_set_small_title_desc': preprocessed_data_set_path + 'valid_set_small_title_desc.txt',

    'mid_data_set_size' : 600000,
    'train_set_mid_only_title': preprocessed_data_set_path + 'train_set_mid_only_title.txt',
    'test_set_mid_only_title': preprocessed_data_set_path + 'test_set_mid_only_title.txt',
    'valid_set_mid_only_title': preprocessed_data_set_path + 'valid_set_mid_only_title.txt',

    'train_set_mid_title_desc': preprocessed_data_set_path + 'train_set_mid_title_desc.txt',
    'test_set_mid_title_desc': preprocessed_data_set_path + 'test_set_mid_title_desc.txt',
    'valid_set_mid_title_desc': preprocessed_data_set_path + 'valid_set_mid_title_desc.txt',

    'train_set_full_only_title': preprocessed_data_set_path + 'train_set_full_only_title.txt',
    'test_set_full_only_title': preprocessed_data_set_path + 'test_set_full_only_title.txt',
    'valid_set_full_only_title': preprocessed_data_set_path + 'valid_set_full_only_title.txt',

    'train_set_full_title_desc': preprocessed_data_set_path + 'train_set_full_title_desc.txt',
    'test_set_full_title_desc': preprocessed_data_set_path + 'test_set_full_title_desc.txt',
    'valid_set_full_title_desc': preprocessed_data_set_path + 'valid_set_full_title_desc.txt',

    # checkpoint
    'checkpoint_dir': checkpoint_dir,

    # cache
    'vocabulary_cache': vocabulary_cache

}
