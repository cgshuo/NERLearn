# -*- coding:utf-8 -*-
# @Time: 2020/9/5 9:21 AM
# @Author: cgshuo
# @Email: cgshuo@163.com
# @File: preprocess.py


import os
import pandas as pd
import re
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import collections

def read_data(fatherpath):
    """
    :param fatherpath: 父文件夹
    :return: corpus, datas, lables, tags

    corpus 句子语料库，{datas,labels}是整合所有数据集的字和其对应label,tags即BOE标签
    -------------labels:--------------
    [['B'], ['B'], ['E'], ['B'], ['E'], ['E'], ['E'], ['E'], ['B']...
    -------------datas:---------------
    [['前'], ['加'], ['大'], ['台'], ['湾'], ['团'], ['员'], ['属']...
    -------------tags:----------------
    {'E', 'O', 'B'}

    """
    pattern = ",\.|/|;|\'|`|\[|\]|<|>|\?|:||\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）"
    labels = list()
    tags = set()
    corpus = list()
    datas = list()
    filenames = os.listdir(fatherpath) #'../ACE2005'
    for filename in filenames:
        with open(os.path.join(fatherpath, filename), 'r') as f:
            sentence = ''
            for line in f.readlines():

                line = line.split()
                linedata = []
                linelabel = []
                numNotO = 0
                linedata.append(line[0])
                for keys in pattern:
                    if line[0] == keys:
                        line[0] = None
                if line[1] != 'O':
                    linelabel.append(line[1])
                    tags.add(line[1])
                else:
                    linelabel.append(line[2])
                    tags.add(line[2])
                # if line[1] != 'O' or line[2] != 'O':
                #     numNotO += 1
                # if numNotO != 0:
                #     datas.append(linedata)
                #     labels.append(linelabel)
                datas.append(linedata)
                labels.append(linelabel)

                sentence += line[0]
            corpus.append(sentence)
    return corpus, datas, labels, tags

def corpus2sentence(corpus):
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
    sentence =[]
    for line in corpus:
        line = filter(None,(re.split(pattern, line)))
        for text in line:
            sentence.append(text)
    return sentence

import collections
def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result
def vocab_build(datas, labels, tags):
    """
    :param datas:
    :param labels:
    :param tags:
    :return:
    """
    all_words = flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words) + 1)

    tags = [i for i in tags]
    tag_ids = range(len(tags))
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)

    word2id["unknow"] = len(word2id) + 1

    return word2id, id2word, tag2id, id2tag

def sentence2id(sent, word2id):
    """
    :param sent: sentence
    :param word2id:
    :return:
    """
    sentenceId = []
    for sentence in sent:
        sentence_id = []
        for word in sentence:
            # if word.isdigit():
            #     word = '<NUM>'
            # elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            #     word = '<ENG>'
            if word not in word2id:
                word = 'unknow'
            sentence_id.append(word2id[word])
        sentenceId.append(sentence_id)
    print(sentenceId)
    return sentenceId

def pad_sequences(sequences, pad_mark=0):
    """
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list

def random_embedding(vocab, embedding_dim):
    """
    :param vocab:# def X_padding(words):
    :param embedding_dim:#     ids = list(word2id[words])
    :return:#         #print(ids)
    """#     if len(ids) >= max_len:
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))#         return ids[:max_len]
    embedding_mat = np.float32(embedding_mat)#     ids.extend([0] * (max_len - len(ids)))
    return embedding_mat#     return ids
#
# def y_padding(tags):
#     ids = list(tag2id[tags])
#     if len(ids) >= max_len:
#         return ids[:max_len]
#     ids.extend([0] * (max_len - len(ids)))
#     return ids

if __name__ == '__main__':
    dic = '../ACE2005'
    corpus, datas, labels, tags = read_data(dic)
    word2id, id2word, tag2id, id2tag = vocab_build(datas, labels, tags)
    with open('../word2id.pkl', 'wb') as outp:
            pickle.dump(word2id, outp)
    # sentence = corpus2sentence(corpus)
    # sentenceId = sentence2id(sentence,word2id)