# -*- coding:utf-8 -*-
# @Time: 2020/9/4 4:13 PM
# @Author: cgshuo
# @Email: cgshuo@163.com
# @File: bilstm_crf_train.py

import pickle
import torch
import torch.optim as optim
from models.BiLSTM_CRF import BiLSTM_CRF
from models.resultCal import calculate

with open('../ACEdata.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)
    x_test = pickle.load(inp)
    y_test = pickle.load(inp)
    x_valid = pickle.load(inp)
    y_valid = pickle.load(inp)
print(x_train)
print(y_train)
print("train len:", len(x_train))
print("test len:", len(x_test))
print("valid len", len(x_valid))

#############
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCHS = 5

tag2id[START_TAG] = len(tag2id)
tag2id[STOP_TAG] = len(tag2id)

model = BiLSTM_CRF(len(word2id) + 1, tag2id, EMBEDDING_DIM, HIDDEN_DIM)

optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)

for epoch in range(EPOCHS):
    index = 0
    for sentence, tags in zip(x_train, y_train):
        index += 1
        model.zero_grad()

        sentence = torch.tensor(sentence, dtype=torch.long)
        tags = torch.tensor([tag2id[t] for t in tags], dtype=torch.long)

        loss = model.neg_log_likelihood(sentence, tags)

        loss.backward()
        optimizer.step()
        if index % 1000 == 0:
            print("epoch", epoch, "index", index)
    entityres = []
    entityall = []
    for sentence, tags in zip(x_test, y_test):
        sentence = torch.tensor(sentence, dtype=torch.long)
        score, predict = model(sentence)
        entityres = calculate(sentence, predict, id2word, id2tag, entityres)
        entityall = calculate(sentence, tags, id2word, id2tag, entityall)
    jiaoji = [i for i in entityres if i in entityall]
    if len(jiaoji) != 0:
        zhun = float(len(jiaoji)) / len(entityres)
        zhao = float(len(jiaoji)) / len(entityall)
        print("test:")
        print("准确率:", zhun)
        print("召回率:", zhao)
        print("F-score:", (2 * zhun * zhao) / (zhun + zhao))
    else:
        print("准确率:", 0)

    # path_name = "./model/model" + str(epoch) + ".pkl"
    # print(path_name)
    # torch.save(model, path_name)
    # print("model has been saved")