'''
part-A: 数据探索
'''
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('\npart-A: 数据探索')

# 将已经分类好的每条评论的路径放进列表
pos_txts = os.listdir('res\\datanew\\pos')
neg_txts = os.listdir('res\\datanew\\neg')

print('num of pos: {0}'.format(len(pos_txts))) # pos样本个数 2000
print('num of neg: {0}'.format(len(neg_txts))) # neg样本个数 2000

# # 随机展示5条pos文本
# pos_samples = random.sample(pos_txts, 5)
# for item in pos_samples:
#     with open(os.path.join('res\\datanew\\pos', item), 'r', encoding='utf-8') as f:
#         s = f.read()
#         print(item)
#         print(s)

# 将所有的评论内容放置到一个list里，列表中的每个元素是一条评论
train_texts_orig = [] 
for i in range(len(pos_txts)):
    with open('res\\datanew\\pos\\'+pos_txts[i], 'r', errors='ignore', encoding='UTF-8') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()
for i in range(len(neg_txts)):
    with open('res\\datanew\\neg\\'+neg_txts[i], 'r', errors='ignore', encoding='UTF-8') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()
print('num of all in list: {0}'.format(len(train_texts_orig))) # list长度 4000

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
print('num of train_tokens: {0}'.format(len(train_tokens))) # 4000

'''
part-C: 数据预处理-索引化
'''
import matplotlib.pyplot as plt
import numpy as np
print('\npart-C: 数据预处理-索引化')

# 获得每条评论的长度，即分词后词语的个数，并将列表转换为ndarray格式
num_tokens = [len(tokens) for tokens in train_tokens]
num_tokens = np.array(num_tokens)

print('max-len of train_tokens: {0}'.format(np.max(num_tokens)))  # 最长评价的长度 1438
print('mean-len of train_tokens: {0}'.format(np.mean(num_tokens)))  # 平均评论的长度 68.77625

# # 绘制评论长度直方图
# plt.hist(np.log(num_tokens), bins = 100)
# plt.xlim((0,10))
# plt.ylabel('num of train_tokens')
# plt.xlabel('len of train_tokens')
# plt.show()

# 每段评语的长度不一，需要将索引长度标准化
mid_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
mid_tokens = int(mid_tokens)
rate = np.sum( num_tokens < mid_tokens ) / len(num_tokens)
print('selected mid-len of train_tokens: {0}'.format(mid_tokens)) # 选取一个平均值，尽可能多的覆盖 227
print('cover rate: {0}'.format(rate)) # 覆盖率 0.956

'''
part-D: 数据预处理-重新构建词向量
'''
print('\npart-D: 数据预处理-重新构建词向量')

print('num of vector: {0}'.format(len(cut_list))) # 预训练的词向量词汇数 255362

# 为了节省训练时间，抽取前50000个词构建新的词向量
num_words = 50000 
embedding_dim = 300

# 初始化embedding_matrix，之后在keras上进行应用
# embedding_matrix为一个 [num_words，embedding_dim] 的矩阵，维度为 50000 * 300
embedding_matrix = np.zeros((num_words, embedding_dim))
for i in range(num_words):
    embedding_matrix[i,:] = cn_model[cn_model.index_to_key[i]]
embedding_matrix = embedding_matrix.astype('float32')

# # 检查新构建的词向量与预训练的词向量index是否对应
# print(cn_model[cn_model.index_to_key[10]])
# print(embedding_matrix[10])
# np.sum(cn_model[cn_model.index_to_key[10]] == embedding_matrix[222] ) # 300

# 新建词向量的维度，keras会用到
embedding_matrix.shape # (50000, 300)

'''
part-E: 数据预处理-填充与裁剪
'''
from keras.preprocessing.sequence import pad_sequences
print('\npart-E: 数据预处理-填充与裁剪')

# 输入的train_tokens是一个list，返回的train_pad是一个numpy array，采用前面（pre）填充的方式
train_pad = pad_sequences(train_tokens, maxlen=mid_tokens, padding='pre', truncating='pre')

# 超出五万个词向量的词用0代替
train_pad[train_pad>=num_words] = 0

train_pad[33] # padding之后前面的tokens全变成0，文本在最后面

# 准备实际输出结果向量向量，前2000好评的样本设为1，后2000差评样本设为0
train_target = np.concatenate((np.ones(2000),np.zeros(2000)))
print(train_target.shape) # (4000,)

'''
part-F: 训练
'''
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
print('\npart-F: 训练')

# 90%的样本用来训练，剩余10%用来测试
X_train, X_test, y_train, y_test = train_test_split(train_pad, train_target, test_size=0.1, random_state=12)

# # 查看训练样本
# # 用索引反向生成语句，索引为零的标记为空格字符
# def reverse_tokens(tokens):
#     text = ''
#     for i in tokens:
#         if i != 0:
#             text = text + cn_model.index_to_key[i]
#         else:
#             text = text + ' '
#     return text
# print(reverse_tokens(X_train[66]))
# print('pred: ',y_train[66])

# 用keras构建顺序模型
model = Sequential()

# 模型第一层为embedding
model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=mid_tokens, trainable=False))

# 循环层使用两层LSTM长短期记忆网络，其中第一层为双向的
model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
model.add(LSTM(units=16, return_sequences=False))

# 定义全连接层，激活函数使用sigmoid
model.add(Dense(1, activation='sigmoid'))

# 梯度 下降使用adam算法，学习率设为0.01
optimizer = Adam(lr=1e-3)

# 定义反向传播，使用交叉熵损失函数，评估函数使用平均值
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 查看模型的结构，一共90k左右可训练的变量
model.summary()

'''
part-G: 调试
'''
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
print('\npart-G: 调试')

# 建立一个权重的存储点，保存训练中的最好模型
path_checkpoint = 'tmp\\weights.hdf5'
checkpointer = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss' , verbose=1 , save_weights_only=True , save_best_only=True)

# 尝试加载已训练模型
try:
    model.load_weights(path_checkpoint)
except Exception as e:
    print(e)

# 定义early stoping如果3个epoch内validation loss没有改善则停止训练
earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# 自动降低learning rate
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)

# 定义callback函数
callbacks = [earlystopping, checkpointer, lr_reduction]

# 开始训练
model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=128, callbacks=callbacks)

# 显示准确率
result = model.evaluate(X_test, y_test)
print('Accuracy:{0:.2%}'.format(result[1]))
