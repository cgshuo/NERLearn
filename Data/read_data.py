# -*- coding:utf-8 -*-
# @Time: 2020/9/4 8:15 PM
# @Author: cgshuo
# @Email: cgshuo@163.com
# @File: read_data.py

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import collections
def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def sentence2id(sent, word2id):
    """
    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        sentence_id.append(word2id[word])
    return sentence_id

def read_data():
    pattern = ",\.|/|;||`|\[|\]|<|>|\?|？ O”“：、':||\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）"
    filepath =[]
    datas = list()
    labels = list()
    linelabel = list()
    tags = set()
    corpus=list()
    filenames = os.listdir('../ACE2005')
    for filename in filenames:
        with open(os.path.join('../ACE2005', filename), 'r') as f:
            sentence =''
            for line in f.readlines():

                line = line.split()
                linedata = []
                linelabel = []
                numNotO = 0

                for keys in pattern:
                    if line[0] in keys:
                        line[0] = ''
                        line[1] = ''
                        line[2] = ''

                linedata.append(line[0])
                if line[1] != 'O':
                     linelabel.append(line[1])
                     tags.add(line[1])
                else:
                    linelabel.append(line[2])
                    tags.add(line[2])
                datas.append(linedata)
                labels.append(linelabel)


            #     linedata.append(line[0])
            #     if line[1] != 'O':
            #         linelabel.append(line[1])
            #         tags.add(line[1])
            #     else:
            #         linelabel.append(line[2])
            #         tags.add(line[2])
            #     if line[1] != 'O' or line[2] != 'O':
            #         numNotO += 1
            #     if numNotO != 0:
            #         datas.append(linedata)
            #         labels.append(linelabel)
            #     sentence += line[0]
            # corpus.append(sentence)


    # print("-------------labels:")
    # print(labels)
    #
    # print("-------------datas:")
    # print(datas)
    #
    # print("-------------tags:")
    # print(tags)
    #
    # print(len(datas), tags)
    # print(len(labels))

    all_words = flatten(datas)
    print(all_words)
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
    max_len = 60

    def X_padding(words):
        ids = list(word2id[words])
        #print(ids)
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    def y_padding(tags):
        ids = list(tag2id[tags])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids


    # df_datas = pd.DataFrame({'sentence': corpus, 'tags': labels}, index=range(len(corpus)))
    # print(df_datas)


    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
    df_data2 = pd.DataFrame({'words': all_words, 'tags': flatten(labels)}, index=range(len(datas)))
    print(df_data2)
    # s = np.asarray(df_data)
    # print(s)

    df_data2.to_csv('./test.csv', index = False, header = False )
    print("完成写入")
    # df_data['x'] = df_data['words'].apply(X_padding)
    # df_data['y'] = df_data['tags'].apply(y_padding)
    #
    # x = np.asarray(list(df_data['x'].values))
    # y = np.asarray(list(df_data['y'].values))
    #
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=43)
    #
    # with open('../ACEdata.pkl', 'wb') as outp:
    #     pickle.dump(word2id, outp)
    #     pickle.dump(id2word, outp)
    #     pickle.dump(tag2id, outp)
    #     pickle.dump(id2tag, outp)
    #     pickle.dump(x_train, outp)
    #     pickle.dump(y_train, outp)
    #     pickle.dump(x_test, outp)
    #     pickle.dump(y_test, outp)
    #     pickle.dump(x_valid, outp)
    #     pickle.dump(y_valid, outp)
    print('** Finished saving the data.')

if __name__ =='__main__':
    read_data()



