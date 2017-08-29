#!/usr/bin/env python
#####################################################
#
# Filename      : cnn/options.py
#
# Author        : JiangTingyu - Jiangty08@gmail.com
# Description   : ---
# Create        : 2017-08-14 10:17:35
# coding=utf-8
#####################################################


class Options(object):
    """Options used by our tensorflow model."""

    def __init__(self):
        # Training options.
        self.batch_size = FLAGS.batch_size

        self.sequence_len = FLAGS.sentence_len

        self.vocab_size = 0
        self.embed_size = FLAGS.embed_size

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

class OptionsTDWC(Options):
    """Options used by TDWC model"""
    def __init__(self):
        self.data_info = None
        self.embed_info = None
        self.input_type_num = None

