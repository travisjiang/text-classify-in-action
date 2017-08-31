#!/usr/bin/env python
import sklearn.datasets

from termcolor import colored
import sys
import os
import glob
import shutil


categories_n = ['rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.space', 'rec.motorcycles', 'misc.forsale']

categories_2 = ['likes', 'dislikes']


def load_files_n_categories(path):

    dir_path = path or 'dataset'

    # remove non utf-8 file
    #remove_incompatible_files(dir_path)

    print '\n\n'

    # load data
    print colored('Loading files into memory', 'green', attrs=['bold'])
    files = sklearn.datasets.load_files(dir_path, categories=categories_n)

    # refine all emails
    print colored('Refining all files', 'green', attrs=['bold'])
    refine_all_emails(files.data)

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


def reorganize_dataset(path):
    likes = ['rec.sport.hockey', 'sci.crypt', 'sci.electronics']
    dislikes = ['sci.space', 'rec.motorcycles', 'misc.forsale']

    folders = glob.glob(os.path.join(path, '*'))
    if os.path.exists(os.path.join(path, 'likes')):
        return
    else:
        # create `likes` and `dislikes` directories
        if not os.path.exists(os.path.join(path, 'likes')):
            os.makedirs(os.path.join(path, 'likes'))
        if not os.path.exists(os.path.join(path, 'dislikes')):
            os.makedirs(os.path.join(path, 'dislikes'))

        for like in likes:
            files = glob.glob(os.path.join(path, like, '*'))
            for f in files:
                parts = f.split(os.sep)
                name = parts[len(parts) - 1]
                newname = like + '_' + name
                shutil.copy(f, os.path.join(path, 'likes', newname))
            #os.rmdir(os.path.join(path, like))

        for like in dislikes:
            files = glob.glob(os.path.join(path, like, '*'))
            for f in files:
                parts = f.split(os.sep)
                name = parts[len(parts) - 1]
                newname = like + '_' + name
                shutil.copy(f, os.path.join(path, 'dislikes', newname))
            #os.rmdir(os.path.join(path, like))


def remove_incompatible_files(dir_path):
    # find incompatible files
    print colored('Finding files incompatible with utf8: ', 'green', attrs=['bold'])
    incompatible_files = find_incompatible_files(dir_path)
    print colored(len(incompatible_files), 'yellow'), 'files found'

    # delete them
    if(len(incompatible_files) > 0):
        print colored('Deleting incompatible files', 'red', attrs=['bold'])
        delete_incompatible_files(incompatible_files)

def delete_incompatible_files(files):
    """
    Deletes the list of files that are passed to it from file system!
    argument `files` is a list of strings. containing absolute or relative pathes
    """
    import os
    for f in files:
        print colored("deleting file:", 'red'), f
        os.remove(f)


def find_incompatible_files(path):
    """
    Finds the filenames that are incompatible with `CountVectorizer`. These files are usually not compatible with UTF8!
    parameter `path` is the absolute or relative path of the training data's root directory.
    returns a list of strings.
    """

    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    files = sklearn.datasets.load_files(path)
    num = []
    for i in range(len(files.filenames)):
        try:
            count_vector.fit_transform(files.data[i:i + 1])
        except UnicodeDecodeError:
            num.append(files.filenames[i])
        except ValueError:
            pass

    return num

def refine_all_emails(file_data):
    """
    Does `refine_single_email` for every single email included in the list
    parameter is a list of strings
    returns NOTHING!
    """

    for i, email in zip(range(len(file_data)), file_data):
        file_data[i] = refine_single_email(email)


def refine_single_email(email):
    """
    Delete the unnecessary information in the header of emails
    Deletes only lines in the email that starts with 'Path:', 'Newsgroups:', 'Xref:'
    parameter is a string.
    returns a string.
    """

    parts = email.split('\n')
    newparts = []

    # finished is when we have reached a line with something like 'Lines:' at the begining of it
    # this is because we want to only remove stuff from headers of emails
    # look at the dataset!
    finished = False
    for part in parts:
        if finished:
            newparts.append(part)
            continue
        if not (part.startswith('Path:') or part.startswith('Newsgroups:') or part.startswith('Xref:')) and not finished:
            newparts.append(part)
        if part.startswith('Lines:'):
            finished = True

    return '\n'.join(newparts)
