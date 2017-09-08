#!/usr/bin/env python
import sklearn.datasets

from termcolor import colored
import sys
import os
import glob
import shutil

import re



def load_files_n_categories(path):

    dir_path = path or 'dataset'

    # remove non utf-8 file
    #remove_incompatible_files(dir_path)

    # load data
    print colored('Loading files into memory', 'green', attrs=['bold'])
    files = sklearn.datasets.load_files(dir_path)

    return files

def remove_stop_words(files, stop_words_path):
    with open(stop_words_path, 'r') as f:
        lines = f.readlines()
        # convert utf-8 to unicode
        stop_words = set([w.strip().decode('utf8') for w in lines])

    r = re.compile(r"\s+")
    for f in files:
        with open(f, 'r') as fin:

            doc = fin.readlines()
            new_doc = []
            for line in doc:
                seg_list = r.split(line)
                seg_list = [w for w in seg_list if w not in stop_words]
                new_line = " ".join(seg_list)
                new_doc.append(new_line)
        with open(f, 'w') as fout:
            fout.write("\n".join(new_doc))

def main():
    stop_words_path = "../dataset/hanlp_stopwords.txt"
    files = glob.glob("../dataset/cn/SogouCCut/C*/*")
    remove_stop_words(files, stop_words_path)

if __name__ == "__main__":
    main()


