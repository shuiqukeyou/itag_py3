import numpy as np
import sys

# oov_word(out of vocabulary work)：超出已有词向量的词汇
# 30000、3076
def load_data(path='math_90k.npz', test_split=0.1, seed=113, num_words=None,
              num_tag=None, start_word=1, oov_word=2, word_index_from=3,
              tag_len=10, start_tag=1, end_tag=2, tag_index_from=1, per=1):

    np.random.seed(seed)  # 生成相同的随机数

    with np.load(path, allow_pickle=True) as f:
        docs = f['brs']  # 文章
        ms = f['ms']  # mask
        tags = f['sfs']  # tag

    # 只加载部分一定比例的数据，防止炸内存
    docs = docs[:int(len(docs)*per)]
    ms = ms[:int(len(ms)*per)]
    tags = tags[:int(len(tags)*per)]

    # brs, sfs, index = share(brs, sfs)
    # num_sfs += index

    indices = np.arange(len(docs))  # 按文章篇数返回一个array对象：0，1，2...len(brs)-1
    np.random.shuffle(indices)  # 随机打乱array对象顺序

    # 通过随机序列进行打乱
    docs = docs[indices]
    ms = ms[indices]
    tags = tags[indices]
    # 装载shared
    with open('shared.txt', 'r') as input_dic:
        # 转换为dict
        dic = eval(input_dic.readline())

    common_words = {}
    original_words = []
    original_tags = []
    unique_words = []
    unique_tags = []
    word_map_dic = {}
    tag_map_dic = {}

    # 如果没有设定词汇编号上限，则读取所有文章，取其中编号最大的词汇作为上限
    if not num_words:
        num_words = max([max(x) for x in docs])
    # 如果没有设定tag编号上限，则读取所有文章，取其中编号最大的tag作为上限
    if not num_tag:
        num_tag = max([max(x) for x in tags])
    print(max([max(x) for x in tags]))
    # 遍历每篇文章
    # 遍历每个文章的每个词，如果某个词的编号大于设定词汇数量上限，将其设为0
    docs = [[w if (w < num_words) else 0 for w in x] for x in docs]

    # filter(判断函数,可迭代对象)：只保留迭代对象中满足判断函数的值
    # py2返回list，py3返回迭代器
    tags = [filter(lambda x: x < num_tag, sf) for sf in tags]  # 排除所有大于tag编号上限的tag
    tags = [list(sf) for sf in tags]

    unk_mask = [[0 if (w == 0) else 1 for w in x] for x in docs]  # 标记出文章中的所有非0词汇的位置（标记为1）
    brs_lens = [len(br) for br in docs]  # 每篇文章的长度list

    print(min(brs_lens), max(brs_lens), sum(brs_lens) * 1.0 / len(brs_lens))  # 输出文章的最短长度、最长长度、平均长度
    # START mask: 0, UNK mask: 0
    ms = [[0] + m for m in ms]  # 将list的list转换为array的list
    ms = [[a * b for a, b in zip(al, bl)] for al, bl in zip(ms, unk_mask)]  # 通过mask和非0词汇计算出需要进行mask的词汇

    sf_lens = [len(list(sf)) for sf in tags]  # 每组tag的长度list
    print(min(sf_lens), max(sf_lens), sum(sf_lens) * 1.0 / len(sf_lens))  # 输出tags的最短长度、最长长度、平均长度

    # 统计所有非0词将其加入到original_words（原始词汇列表中）
    for x in docs:
        for w in x:
            if w == 0:
                continue
            original_words.append(w)
    # 统计所有非0tag将其加入到original_tags（原始tag列表中）
    for x in tags:
        for t in x:
            original_tags.append(t)

    # 去重
    original_words = list(set(original_words))
    original_tags = list(set(original_tags))


    # 读取shared的所有key
    keys = dic.keys()
    # 遍历所有原始词汇列表
    for w in original_words:
        # 如果某个词汇在shared的key中
        if w in keys:
            # 读取这个key对应的value（tag）
            t = dic[w]
            # 如果读取的标签存在于original_tags中
            if t in original_tags:
                # 将这个词-tag对放入公共词汇dict（common_words）中
                common_words[w] = t
                continue
        # 否则将其放入独立词汇list（unique_words）
        unique_words.append(w)

    # 读取所有公共词汇dict对应的tag
    values = common_words.values()
    for t in original_tags:
        # 如果有从来没有对应词出现的tag
        if t not in values:
            # 将其放入到独立tag list中
            unique_tags.append(t)

    index = 3
    # 遍历所有独立词汇，将其作为key储存到word_map_dict中，value（序号）递增（0~2为空）
    for uw in unique_words:
        word_map_dic[uw] = index
        index += 1

    # 记录tag序号开始的位置
    tag_from = index
    # 共享序号
    shared_index = index
    # 取出所有共同词汇，将其作为key储存到word_map_dic中，value（序号）继续递增
    for w in common_words.keys():
        word_map_dic[w] = index
        index += 1

    # 取出所有共同词汇dict中的tag，将其作为key储存到tag_map_dic中，value（序号）从共享序号处开始递增
    for w in common_words.values():
        tag_map_dic[w] = shared_index
        shared_index += 1

    # 遍历所有独立tag，将其作为key储存到tag_map_dic中，value（序号）继续递增
    for ut in unique_tags:
        tag_map_dic[ut] = shared_index
        shared_index += 1

    #
    sfs_end = shared_index
    sfs_start = shared_index + 1

    # dict.get(key,default)
    # 当获取到文章中不存在于word_map_dic中的词汇的时候，将其默认值设定为2（OOV word）
    # new_brs每个元素的格式为[1,序号]
    new_brs = np.array([[start_word] + [word_map_dic.get(w, oov_word) for w in x] for x in docs])
    # 转换tag_dict为序号array
    new_sfs = np.array([[tag_map_dic[t] for t in x] for x in tags])

    # 划分测试集
    split_index = int(len(docs) * test_split)

    # 按每篇指定上限保留tag
    new_sfs = [sf[:tag_len - 1] for sf in new_sfs]

    # tag_in：[1,tag1,...]（1为起始标记）
    # tag_out：[tag1,tag2,...,2]（2为终止标记）
    sfs_in = [[sfs_start] + sf for sf in new_sfs]
    sfs_out = [sf + [sfs_end] for sf in new_sfs]



    shared_map = open('shared_map.txt', 'w')
    shared_map.write(str(word_map_dic) + '\n')
    shared_map.write(str(tag_map_dic))
    print('unique_words:', len(unique_words))
    print('common:', len(common_words))
    print('unique_tags:', len(unique_tags))
    print('index: ', index)
    print('tags index:', index - len(common_words))
    print('shared_index:', shared_index)
    print('tag from :', tag_from)
    print('sfs end:', sfs_end)
    print('sfs start:', sfs_start)

    return (new_brs[:-split_index], ms[:-split_index], sfs_in[:-split_index], sfs_out[:-split_index]), \
           (new_brs[-split_index:], ms[-split_index:], sfs_in[-split_index:], sfs_out[-split_index:]),tag_from


def encode_one_hot(int_data, vocab_size):
    one_hots = []
    for value in int_data:
        # print('value:', value)
        d = np.zeros(vocab_size, dtype=np.int16)  # 省内存
        if value > 1:
            d[value] = 1
        one_hots.append(d)
    return one_hots

# 输入值：文章、输出tags
# 对于每个文章-tag对，都输出一组one-hot list
def weight_one_hot(words, tags):
    one_hots = []
    for value in tags:
        d = []
        for w in words:
            # 读取文章中的每一个词，如果这个词和tag相同，则将其标记为1，否则标记为0
            if w == value:
                d.append(1)
            else:
                d.append(0)
        one_hots.append(d)
    return one_hots

if __name__ == '__main__':
    # for google drive
    # import os
    # os.chdir("./drive/My Drive/iTag-o")
    load_data("data.npz", num_words=20000, num_tag=3073, per=0.5)