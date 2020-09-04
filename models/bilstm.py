# -*- coding:utf-8 -*-
# @Time: 2020/9/4 10:39 AM
# @Author: cgshuo
# @Email: cgshuo@163.com
# @File: bilstm.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """
        :param vocab_size: 字典大小
        :param emb_size: 词向量维数
        :param hidden_size: 隐向量的维数
        :param out_size: 标注的种类
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        #一个简单的查找表，用于存储固定字典和大小的嵌入。
        #该模块通常用于存储单词嵌入并使用索引检索它们。 模块的输入是索引列表，输出是相应的词嵌入

        self.bilstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        #batch_first –如果为True，则输入和输出张量按(batch，seq，feature）提供。 默认值：False

        self.lin = nn.Linear(in_features=2*hidden_size, out_size=out_size)
        #in_features –每个输入样本的大小 out_features –每个输出样本的大小
        #对输入数据应用线性变换：y=xAT+b

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor) #  [B, L, emb_size]

        packed = pack_padded_sequence(input=emb, lengths=lengths, batch_first=True)
        #打包一个 Tensor，其中包含可变长度的填充序列。
        #input的大小可以为T x B x *，其中 <cite>T</cite> 是最长序列的长度(等于lengths[0]），
        # B是批处理大小，*是任意数量的尺寸 (包括 0）。 如果batch_first为True，则预期为B x T x * input。

        rnn_out, _ = self.bilstm(packed)
        #rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        scores = self.lin(rnn_out)  # [B, L, out_size]

        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids