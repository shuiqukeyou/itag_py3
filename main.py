# -*- coding=utf8 -*-
import os
import sys

import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, TimeDistributed, Dense, Dropout, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping

from function import mean_negative_log_probs, compute_precision, compute_recall
from dataload import load_data, weight_one_hot, encode_one_hot
from config import ALL_WORDS, WORD_VOCAB, LABEL_VOCAB, DE_TOKENS, MAX_WORDS, MAX_LABELS, GRU_SIZE, ATTENTION_SIZE, \
    EMBEDDING_DIM, KEEP_PROB, NUM_EPOCHS, BATCH_SIZE, START_TOKEN, DATA_FILE, LABEL_FROM, END_TOKEN, BEAM_SIZE, \
    MAX_LENGTH, PER

from layers import Masked, AttentionLayer

# for google drive
# os.chdir("./drive/My Drive/itag_py3")

# set use gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 模型搭建
# 编码输入层（输入词汇）（最大允许100）
encoder_input = Input(shape=(MAX_WORDS,))
# 解码输入层（输入标签）（最大允许5）
decoder_input = Input(shape=(MAX_LABELS,))

# 公共embedding层，标签-内容的语义一致性实现
# 编码尺寸：20705 * 100（字典长度*嵌入维度）
shared_embedded = Embedding(ALL_WORDS, EMBEDDING_DIM, mask_zero=True)
# 将输入序列进行编码
encoder_embedded = shared_embedded(encoder_input)


# 编码器部分三层GRU，前两层使用dropout，最后一层使用一个自定义的随机失活，同时调整输出的尺寸
# return_sequences=True：返回每个时间步的隐藏状态,传往下一层
# return_state=True：返回最后一个时间步的隐藏状态，传往解码器
encoder_gru1 = GRU(GRU_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal',
                   recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=KEEP_PROB)
print(0, encoder_embedded.shape)

encoder_outputs, state1 = encoder_gru1(encoder_embedded)

encoder_gru2 = GRU(GRU_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal',
                   recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=KEEP_PROB)
encoder_outputs, state2 = encoder_gru2(encoder_outputs)
encoder_gru3 = GRU(GRU_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal',
                   recurrent_initializer='orthogonal', bias_initializer='zeros')
encoder_outputs, state3 = encoder_gru3(encoder_outputs)

# 自定义的mask层(?,?256) → (?,100,256)
encoder_outputs = Masked()(encoder_outputs)

# 将解码输入层通过公共embedding层进行嵌入
decoder_outputs = shared_embedded(decoder_input)

# 解码器部分三层GRU，前两层使用dropout, 每层都使用对应编码层的状态进行初始化
decoder_gru1 = GRU(GRU_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal',
                   recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=KEEP_PROB)
decoder_outputs, n_state = decoder_gru1(decoder_outputs, initial_state=state1)
decoder_gru2 = GRU(GRU_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal',
                   recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=KEEP_PROB)
decoder_outputs, n_state = decoder_gru2(decoder_outputs, initial_state=state2)
decoder_gru3 = GRU(GRU_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal',
                   recurrent_initializer='orthogonal', bias_initializer='zeros')
decoder_outputs, n_state = decoder_gru3(decoder_outputs, initial_state=state3)

attention = AttentionLayer(units=ATTENTION_SIZE, return_alphas=True)
decoder_outputs, decoder_alphas, decoder_pgen = attention([encoder_outputs, decoder_outputs])
print(decoder_outputs.shape)
# 添加dropout
decoder_outputs = Dropout(KEEP_PROB)(decoder_outputs)

# 解码连接层
decoder_dense = Dense(DE_TOKENS, activation='softmax')  # tag数+1（终止标记）
# 权重全连接层
weight_dense = Dense(MAX_WORDS, activation='softmax')  # 100

pgen_dense = Dense(1, activation='softmax')

# 解码器输出
y_ = decoder_dense(decoder_outputs)
# 注意力权重输出
w_ = weight_dense(decoder_alphas)
# 指示函数输出
p_ = pgen_dense(decoder_pgen)

model = Model(inputs=[encoder_input, decoder_input], outputs=[y_, w_])
# compile model
model.compile(optimizer='adam', loss=mean_negative_log_probs, metrics=[compute_precision, compute_recall])

model.summary()

# load data
# 文章（处理后）、mask、标签输入、标签输出
# 标签输入：[START,tag1,...]（1为起始标记）
# 标签输出：[tag1,tag2,...,END]（2为终止标记）
(en_train, ms_train, de_train, y_train), (en_test, ms_test, de_test, y_test), tag_from = \
    load_data(path=DATA_FILE, num_words=WORD_VOCAB, num_tag=LABEL_VOCAB, start_tag=LABEL_VOCAB, end_tag=END_TOKEN,
              tag_len=MAX_LABELS, per=PER)

LABEL_FROM = tag_from
# pad_sequences：统一序列长度：
# padding：对不足的序列进行填充。pre：前端填充，post：后端填充，默认填充0.0
# truncating：对超长的序列进行剪切


# print(de_train.shape)

en_train = pad_sequences(en_train, padding='post', truncating='post', maxlen=MAX_WORDS)
de_train = pad_sequences(de_train, padding='post', truncating='post', maxlen=MAX_LABELS)
y_train = pad_sequences(y_train, padding='post', truncating='post', maxlen=MAX_LABELS)

w_train = np.array([weight_one_hot(en_train[i], y_train[i]) for i in range(len(y_train))])


# y_train.shape = (144611, 5)
y_train = [encode_one_hot(y - LABEL_FROM, DE_TOKENS) for y in y_train]
# print(y_train.shape)
y_train = np.array(y_train)

# y_temp = [encode_one_hot(y - LABEL_FROM, DE_TOKENS) for y in y_train]
# y_train = np.array(y_temp.pop(0))  # 创建初始序列
# while(len(y_temp) > 0):
#     temp = y_temp.pop(0)
#     y_train = np.concatenate((y_train, temp), axis=0)


# 开始训练
es = EarlyStopping(monitor='val_loss', patience=2)
cp = ModelCheckpoint(filepath='itag.h5', monitor='val_loss', save_best_only=True)
print(en_train.shape)
print(de_train.shape)
print(y_train.shape)
print(w_train.shape)
model.fit([en_train, de_train],
          [y_train, w_train], validation_split=0.1, epochs=NUM_EPOCHS,
          batch_size=BATCH_SIZE, callbacks=[es, cp], verbose=2)

model.load_weights('itag.h5')  # 加载最好的训练结果


# 创建测试集
en_test = pad_sequences(en_test, padding='post', truncating='post', maxlen=MAX_WORDS)
de_test = pad_sequences(de_test, padding='post', truncating='post', maxlen=MAX_LABELS)
y_test = pad_sequences(y_test, padding='post', truncating='post', maxlen=MAX_LABELS)
yo_test = np.array([encode_one_hot(y - LABEL_FROM, DE_TOKENS) for y in y_test])

# 预测
# 编码器模型，输入：文章，输出：编码序列，三层的隐藏状态
# 编码器权重此时已经训练完毕
encoder_model = Model([encoder_input], [encoder_outputs, state1, state2, state3])

# 创建输入层（用来接编码器输出的三层隐藏状态）
en_state1 = Input(shape=(GRU_SIZE,))
en_state2 = Input(shape=(GRU_SIZE,))
en_state3 = Input(shape=(GRU_SIZE,))
# 100*5
de_context = Input(shape=(MAX_WORDS, GRU_SIZE,))


current_token = Input(shape=(1,))  # 正确输入（单个tag输入）
decoder_out = shared_embedded(current_token)  # 对正确输入进行嵌入
# 三层 GRU
decoder_out, de_state1 = decoder_gru1(decoder_out, initial_state=en_state1)
decoder_out, de_state2 = decoder_gru2(decoder_out, initial_state=en_state2)
decoder_out, de_state3 = decoder_gru3(decoder_out, initial_state=en_state3)

decoder_out, decoder_al, decoder_pg = attention([de_context, decoder_out])

decoder_out = TimeDistributed(decoder_dense)(decoder_out)  # 对所有解码器输出都调用decoder_dense全连接层进行处理
decoder_al = weight_dense(decoder_al)  # 注意力权重
decoder_pg = pgen_dense(decoder_pg)  # 指示函数权重

# 输入：正确的tag、文本编码、三层隐藏状态
# 输出：解码器输出（已softmax）、注意力权重、指示函数权重、解码器三层隐藏状态
decoder_model = Model([current_token, de_context, en_state1, en_state2, en_state3],
                      [decoder_out, decoder_al, decoder_pg, de_state1, de_state2, de_state3])


# 输入：文本原文、本次用于预测的tag（初始为START）、文本编码序列、三层编码器隐藏状态、预测深度（惩罚用）、累积概率、之前的预测结果、tag
def predict_next_token(en, current, full_context, en_st1, en_st2, en_st3, cur_depth, joint_prs, res, tags):
    cur_depth += 1
    # 解码器输出（已softmax）、注意力权重、指示函数权重、解码器三层隐藏状态
    prs, weights, pgen, en_st1, en_st2, en_st3 = decoder_model.predict([current, full_context, en_st1, en_st2, en_st3])

    prs = prs[0, 0, :]

    new_prs = []
    # 对每个解码器输出进行指示函数标记
    for pr in prs:
        new_prs.append(pr * pgen)
    # 如果正文中的一个词在共通范围内
    for i in range(len(en)):
        if (en[i] - LABEL_FROM) > 0 and (en[i] - LABEL_FROM) < DE_TOKENS:
            # 令这个词的标签
            new_prs[en[i] - LABEL_FROM] += weights[0][0][i] * (1 - pgen)
    prs = new_prs
    # xrange：和range一样，但返回生成器，仅py2
    # prs = [(i + 2, v) for i, v in zip(xrange(len(prs)), prs)]

    # 对输出概率进行编号和排序（0开始）
    prs = [(i, v) for i, v in zip(range(len(prs)), prs)]
    # 根据概率进行排序
    # prs = sorted(prs, lambda x, y: cmp(x[1], y[1]) / cur_depth, reverse=True)
    prs.sort(key=lambda p: p[1], reverse=True)

    # 波束搜索
    for p in prs[:BEAM_SIZE]:  # 取波束窗口宽度的前x个
        # if p[0] == END_TOKEN:  # 如果到达了预测末尾
        #     res.append(((joint_prs + p[1]) / cur_depth, tags[:] + [p[0]]))  # res.append(概率值,tag array)
        if cur_depth == MAX_LENGTH:  # 如果深度到了上限且本次预测结果没有出现过
            if p[0] not in tags:
                # 深度加成只在最后得出结果时进行
                res.append(((joint_prs + p[1]) / cur_depth, tags[:] + [p[0]]))  # res.append(概率值,tag array)
        else:
            if p[0] not in tags:
                token = np.zeros((1, 1))
                token[0, 0] = p[0] + LABEL_FROM  # 将输出结果转换为下轮输入
                predict_next_token(en, token, full_context, en_st1, en_st2, en_st3,
                                   cur_depth, joint_prs + np.log(p[1]), res, tags[:] + [p[0]])
        if cur_depth == MAX_LENGTH:
            break

count = 0  # 预测样本计数
recall = 0
precise = 0
full_hit_count = 0

for (en, y) in zip(en_test, y_test):
    count += 1
    context, en_state1, en_state2, en_state3 = encoder_model.predict(np.array([en]))  # 输入文本，产生隐藏序列和三个GRU状态

    # 创建启动tag：START
    cur_token = np.zeros((1, 1))
    cur_token[0, 0] = START_TOKEN

    # 预测tag list
    results = []  # 内容为(tag序列,对数概率)
    predict_next_token(en, cur_token, context, en_state1, en_state2, en_state3, 0, 0.0, results, [])

    # results = sorted(results, lambda x, y: cmp(x[0], y[0]), reverse=True)  # 对输出结果进行排序
    results.sort(key=lambda r: r[0], reverse=True)  # 根据概率对数和进行排序

    if len(results) == 0:
        continue
    decoder_seq = results[0][1]  # 取对数概率最大的输出结果
    decoder_seq = [w + LABEL_FROM for w in decoder_seq]  # 转换到标准词空间中

    # 计算单次精确度和召回率
    y = list(y)
    if count % 1000 == 0:
        print(count)
        print(decoder_seq, y)
    tmp_precision = 0
    tmp_recall = 0

    # 取输出序列和测试集的交集
    intersection = list(set(y).intersection(set(decoder_seq)))
    if END_TOKEN in intersection:
        intersection.remove(END_TOKEN)  # 移除end标记
    y_set = set(y)

    # 删除y的第一项，可能为开始符号（然而并没有）
    # if 0 in y_set:
    #     y_set.remove(0)

    # 移除掉en标记
    if END_TOKEN in y_set:
        y_set.remove(END_TOKEN)
    # 计算召回
    if len(intersection) > 0:
        tmp_recall = len(intersection) * 1.0 / len(y_set)
        recall += tmp_recall

    decoder_seq_set = set(decoder_seq)
    if END_TOKEN in decoder_seq_set:
        decoder_seq_set.remove(END_TOKEN)

    # 计算精确度
    if len(intersection) > 0:
        tmp_precision = len(intersection) * 1.0 / len(decoder_seq_set)
        # precise += len(intersection) * 1.0 / 5
        precise += tmp_precision

    tmp_f1 = 0
    if tmp_recall != 0 or tmp_precision != 0:
        tmp_f1 = 2 * tmp_precision * tmp_recall / (tmp_precision + tmp_recall)

    # 计算F1
    isHit = True
    for d, yl in zip(decoder_seq, y):
        if d != yl:
            isHit = False
            break
    if isHit:
        full_hit_count += 1

full_hit_count /= len(en_test) * 1.0
precise /= len(en_test) * 1.0
recall /= len(en_test) * 1.0
f1score = 2 * precise * recall / (precise + recall)
print("full hit: %f, precision@%d: %f, recall@%d: %f, f1@%d: %f" % (full_hit_count, MAX_LENGTH, precise, MAX_LENGTH, recall, MAX_LENGTH, f1score))