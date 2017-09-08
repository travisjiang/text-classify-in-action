#!/usr/bin/env python
# coding=utf-8



import sklearn.datasets

from termcolor import colored
import os
import glob
import shutil

import jieba

import sys
reload(sys)
sys.setdefaultencoding('utf8')


categories_n = ['Auto', 'Culture', 'Economy', 'Medicine', 'Military', 'Sports']





def load_files_n_categories(path):

    dir_path = path or 'dataset'

    # remove non utf-8 file
    #remove_incompatible_files(dir_path)

    print '\n\n'

    # load data
    print colored('Loading files into memory', 'green', attrs=['bold'])
    files = sklearn.datasets.load_files(dir_path, categories=categories_n)

    return files

def load_files_2_categories(path):

    dir_path = path

    # preprocess data into two folders instead of 6
    print colored("Reorganizing folders, into two classes", 'cyan', attrs=['bold'])
    reorganize_dataset(path)


    #remove_incompatible_files(dir_path)

    print '\n\n'

    # load data
    print colored('Loading files into memory', 'green', attrs=['bold'])
    files = sklearn.datasets.load_files(dir_path, categories=categories_2)

    # refine all emails
    print colored('Refining all files', 'green', attrs=['bold'])
    refine_all_emails(files.data)

    return files


def clean_dataset(path, input_dir, output_dir):
    if not os.path.exists(os.path.join(path, output_dir)):
        os.makedirs(os.path.join(path, output_dir))

    files = glob.glob(os.path.join(path, input_dir, "*"))
    for f in files:
        f_name = f.split(os.sep)[-1]

        output_file = os.path.join(path, output_dir, f_name)

        # 繁简转换
        shell_opencc = "opencc -i %s -o %s -c zht2zhs.ini" % (f, output_file)
        os.system(shell_opencc)

        # 去除非utf-8字符，不太必要，可设置解码时忽略即可
        shell_iconv = "iconv -c -t UTF-8 < %s > %s" % (output_file, output_file)
        os.system(shell_iconv)

def segment_dataset(path, input_dir, output_dir, stop_words_path=None):
    if not os.path.exists(os.path.join(path, output_dir)):
        os.makedirs(os.path.join(path, output_dir))
    stop_words = set()
    try:
        with open(stop_words_path, 'r') as f:
            lines = f.readlines()
            # convert utf-8 to unicode
            stop_words = set([w.strip().decode('utf8') for w in lines])
    except IOError as err:
        print("File Error:"+str(err))
    except TypeError as err:
        print("File Error:"+str(err))
    print(stop_words)

    files = glob.glob(os.path.join(path, input_dir, "*"))
    for f in files:
        with open(f, 'r') as fin:
            f_name = f.split(os.sep)[-1]

            with open(os.path.join(path, output_dir, f_name), 'w') as fout:
                doc = fin.readlines()
                for line in doc:
                    # jieba will convert utf8 to unicode
                    seg_list = jieba.cut(line)
                    seg_list = [w for w in seg_list if w not in stop_words]
                    new_line = " ".join(seg_list)
                    fout.write(new_line)




def reorganize_dataset(path, input_dir):
    files = glob.glob(os.path.join(path, input_dir, "*"))
    for f in files:
        f_name = f.split(os.sep)[-1]
        name_parts = f_name.split('_')
        prefix, _ = name_parts[0], name_parts[1]
        if not os.path.exists(os.path.join(path, prefix)):
            os.makedirs(os.path.join(path, prefix))
        shutil.copy(f, os.path.join(path, prefix, f_name))

def main():
    path = "../dataset/cn/netease"
    #stop_words_path = "../dataset/hanlp_stopwords.txt"
    stop_words_path = None

    #clean_dataset(path, "origin", "cleaned")

    segment_dataset(path, "origin", "cutted", stop_words_path)

    reorganize_dataset(path, "cutted")

if __name__ == "__main__":
    main()





