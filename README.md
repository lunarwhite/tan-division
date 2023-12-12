# tan-division

![GitHub Repo stars](https://img.shields.io/github/stars/lunarwhite/tan-division?color=orange)
![GitHub watchers](https://img.shields.io/github/watchers/lunarwhite/tan-division?color=yellow)
![GitHub forks](https://img.shields.io/github/forks/lunarwhite/tan-division?color=green)
![GitHub top language](https://img.shields.io/github/languages/top/lunarwhite/tan-division)
![GitHub License](https://img.shields.io/github/license/lunarwhite/tan-division?color=white)

Try Chinese corpus sentiment analysis with TensorFlow + Keras.

```
├───.gitignore 
├───README.md
├───requirements.txt
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

## 1 Overview

- 基于谭松波老师的酒店评论数据集的中文文本情感分析，二分类问题
- 数据集标签有 `pos` 和 `neg`，分别 2000 条 txt 文本
- 选择 RNN、LSTM 和 Bi-LSTM 作为基础模型，借助 Keras 搭建训练
- 主要工具包版本为 TensorFlow `2.0.0`、Keras `2.3.1` 和 Python `3.6.2`
- 在测试集上可稳定达到 92% 的准确率

## 2 Setup

- clone repo：`git clone https://github.com/lunarwhite/tan-division.git`
- 更新 pip：`pip3 install --upgrade pip`
- 为项目创建虚拟环境：`conda create --name <env_name> python=3.6`
- 激活 env：`conda activate <env_name>`
- 安装 Python 库依赖：`pip3 install -r requirements.txt`
- 下载封装好的[中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)，本项目选取 [Zhihu_QA Word + Ngram](https://pan.baidu.com/s/1OQ6fQLCgqT43WTwh5fh_lg)，放在 `res/word-vector` 路径下

## 3 Train

- 运行：`python src/run.py`
- 调参：在 `src/run.py` 文件中修改常用参数，如下
    ```python
    my_lr = 1e-2 # 初始学习率
    my_test_size = 0.1
    my_validation_split = 0.1 # 验证集比例
    my_epochs = 40 # 训练轮数
    my_batch_size = 128 # 批大小
    my_dropout = 0.2 # dropout 参数大小
    
    my_optimizer = Nadam(lr=my_lr) # 优化方法
    my_loss = 'binary_crossentropy' # 损失函数
    ```

## 4 Workflow

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
