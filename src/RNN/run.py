'''
part-A: 数据探索
'''
import os
print('\npart-A: 数据探索')

# 将已经分类好的每条评论的路径放进列表
pos_txts = os.listdir('res\\datanew\\pos')
neg_txts = os.listdir('res\\datanew\\neg')

print('num of pos: {0}'.format(len(pos_txts))) # pos样本个数为2000
print('num of neg: {0}'.format(len(neg_txts))) # neg样本个数为2000

# 将所有的评论内容放置到一个list里，列表中的每个元素是一条评论
train_texts_orig = [] 
for i in range(len(pos_txts)):
    with open('res\\datanew\\pos\\'+pos_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()
for i in range(len(neg_txts)):
    with open('res\\datanew\\neg\\'+neg_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()
print('num of all in list: {0}'.format(len(train_texts_orig))) # list长度应该为4000

'''
part-B: 数据预处理-分词
'''
import re
import jieba
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
print('\npart-B: 数据预处理-分词')

# 使用gensim加载已经训练好的汉语词向量
cn_model = KeyedVectors.load_word2vec_format('res\\word-vector\\sgns.zhihu.bigram.bz2', binary=False)

# 用jieba进行中文分词，最后将每条评论转换为了词索引的列表
train_tokens = []
for text in train_texts_orig:
    # 对每条评论进行去标点符号处理
    text = re.sub("[\s+\.\!\/_,-|$%^*(+\"\')]+|[+——！，； 。？ 、~@#￥%……&*（）]+", "", text)
    cut = jieba.cut(text)
    cut_list = [i for i in cut]
    for i, word in enumerate(cut_list):
        try:
            # 将分出来的每个词转换为词向量中的对应索引
            cut_list[i] = cn_model.key_to_index[word]
        except KeyError:
            # 如果词不在词向量中，则索引标记为0
            cut_list[i] = 0
    train_tokens.append(cut_list)
print('num of train_tokens: {0}'.format(len(train_tokens)))

'''
part-C: 数据预处理-索引化
'''
import matplotlib.pyplot as plt
import numpy as np
print('\npart-C: 数据预处理-索引化')

# 获得每条评论的长度，即分词后词语的个数，并将列表转换为ndarray格式
num_tokens = [len(tokens) for tokens in train_tokens]
num_tokens = np.array(num_tokens)
print(num_tokens)

np.max('num of train_tokens: {0}'.format(num_tokens)) # 最长的评价的长度
np.mean('num of train_tokens: {0}'.format(num_tokens)) # 平均评论的长度

# 每段评语的长度不一，需要将索引长度标准化



# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import *
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
# from sklearn.model_selection import train_test_split