# Chinese-corpus-sentiment-data-analysis
 
![Chinese-corpus-sentiment-data-analysis](https://socialify.git.ci/lunarwhite/Chinese-corpus-sentiment-data-analysis/image?description=1&descriptionEditable=ML-based%20sentiment-analysis%20of%20simple%20Chinese%20corpus.%20Pick%20models%20RNN%2C%20LSTM%20and%20Bi-LSTM.%20Use%20%0A%20Keras%20and%20Tensorflow.&font=Raleway&forks=1&issues=1&language=1&logo=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2Fa%2Fae%2FKeras_logo.svg%2F180px-Keras_logo.svg.png&owner=1&pattern=Overlapping%20Hexagons&pulls=1&stargazers=1&theme=Light)

```
├───.gitignore 
├───README.md
├───requirements.txt
├───.vscode
│    └── settings.json  
├───res
│   ├───datanew
│   │   ├───neg
│   │   └───pos
│   └───word-vector
│       └───sgns.zhihu.bigram.bz2
├───src
│   └───run.py
└───tmp
    └───weights.hdf5
```

## 1 概览
- 基于谭松波老师的酒店评论数据集的中文文本情感分析，二分类问题
- 数据集标签有pos和neg，分别2000条txt文本
- 选择RNN、LSTM和Bi-LSTM作为模型，借助Keras搭建训练
- 主要工具包版本为TensorFlow 2.0.0、Keras 2.3.1和Python 3.6.2
- 在测试集上可稳定达到92%的准确率

## 2 部署
- 克隆repo：`git clone https://github.com/lunarwhite/Chinese-corpus-sentiment-data-analysis.git`
- 更新pip：`pip3 install --upgrade pip`
- 为项目创建虚拟环境：`conda create --name <env_name> python=3.6`
- 激活env：`conda activate <env_name>`
- 安装python库依赖：`pip3 install -r requirements.txt`
- 下载封装好的[中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)，本项目选择的是[Zhihu_QA Word + Ngram](https://pan.baidu.com/s/1OQ6fQLCgqT43WTwh5fh_lg)，并放在`res/word-vector`目录下

## 3 运行
- 运行：`python src/run.py`
- 调参：在`src/run.py`文件中修改常用参数，如下
    ```python
    my_lr = 1e-2 # 初始学习率
    my_test_size = 0.1
    my_validation_split = 0.1 # 验证集比例
    my_epochs = 40 # 训练轮数
    my_batch_size = 128 # 批大小
    my_dropout = 0.2 # dropout参数大小
    
    my_optimizer = Nadam(lr=my_lr) # 优化方法
    my_loss = 'binary_crossentropy' # 损失函数
    ```

## 4 流程
- 观察数据
  - 数据集大小
  - 数据集样本
  - 样本长度
- 数据预处理
  - 分词
  - 短句补全、长句裁剪
  - 索引化
  - 构建词向量
- 搭建模型
  - RNN
  - LSTM
  - Bi-LSTM
- 可视化分析
  - epochs-loss
  - epochs-accuracy
- 调试
  - callback
  - checkpoint
- 改进模型
  - loss function
  - optimizer
  - learning rate
  - epochs
  - batch_size
  - dropout
  - early-stopping
