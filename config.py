# -*- coding=utf8 -*-

# 全词汇表尺寸(文本+tag)
ALL_WORDS = 35000
# 词one-hot向量长度（文本词汇总量）
WORD_VOCAB = 20000
# 标签one-hot向量长度（标签词汇总量）
LABEL_VOCAB = 3076
# 删除掉填充符号和起始符号后的标签热向量
DE_TOKENS = LABEL_VOCAB + 1  # remove pad and start tokens from file_vocab

# 单次输入模型的最大词汇量
MAX_WORDS = 100
# 输出最大标签量
MAX_LABELS = 5

# GRU层尺寸、注意力层尺寸、嵌入层长度
GRU_SIZE = 256
ATTENTION_SIZE = 256
EMBEDDING_DIM = 100

# dropout丢弃比例
KEEP_PROB = 0.1
# 迭代次数
NUM_EPOCHS = 60
# mini-batch尺寸
BATCH_SIZE = 100

# 未使用
INDEX_FROM = 3
# 结束位置序号

END_TOKEN = 30951
# 起始位置序号
START_TOKEN = 31250
LABEL_FROM = 27878

# 未使用
TOPIC_NUM = 100
# 波束搜索尺寸
BEAM_SIZE = 3
# tags最大长度
MAX_LENGTH = 5

PER = 0.05

# 数据集名
DATA_FILE = "data.npz"