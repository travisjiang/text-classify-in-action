#!/usr/bin/env python
import sklearn.datasets

from termcolor import colored
import sys
import os
import glob
import shutil


def load_files_n_categories(path):

    dir_path = path or 'dataset'

    # remove non utf-8 file
    #remove_incompatible_files(dir_path)

    print '\n\n'

    # load data
    print colored('Loading files into memory', 'green', attrs=['bold'])
    files = sklearn.datasets.load_files(dir_path)

    return files

