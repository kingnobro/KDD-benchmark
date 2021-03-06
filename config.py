#!/usr/bin/env python
# encoding: utf-8
import os
import socket

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

# 当前工作目录
CWD = "/Users/apple/Desktop/KDD_Benchmark"  # Linux系统目录
# CWD = "D:\KDD\KDD_Benchmark" # Windows系统目录

DATA_PATH = os.path.join(CWD, "data")
DATASET_PATH = os.path.join(DATA_PATH, "dataset")

# 训练和测试文件（训练阶段用验证数据，测试阶段使用测试数据）
TRAIN_FILE = os.path.join(DATASET_PATH, "train_set", "Train.csv")
# TEST_FILE = os.path.join(DATASET_PATH, "valid_set", "Valid.csv")      # 验证集
TEST_FILE = os.path.join(DATASET_PATH, "test_set", "Test.01.csv")       # 预测集
GOLD_FILE = os.path.join(DATASET_PATH, "valid_set", "Valid.gold.csv")

# 模型文件
MODEL_PATH = os.path.join(CWD, "model", "kdd.model")
# 训练和测试特征文件
TRAIN_FEATURE_PATH = os.path.join(CWD, "feature", "train.feature")
TEST_FEATURE_PATH = os.path.join(CWD, "feature", "test.feature")
# 分类在测试集上的预测结果
TEST_RESULT_PATH = os.path.join(CWD, "predict", "test.result")
# 重新格式化的预测结果
TEST_PREDICT_PATH = os.path.join(CWD, "predict", "test.predict")

COAUTHOR_FILE = os.path.join(DATASET_PATH, "coauthor.json")
PAPERIDAUTHORID_TO_NAME_AND_AFFILIATION_FILE = os.path.join(DATASET_PATH,
                                                            "paperIdAuthorId_to_name_and_affiliation.json")
PAPERAUTHOR_FILE = os.path.join(DATASET_PATH, "PaperAuthor.csv")
AUTHOR_FILE = os.path.join(DATASET_PATH, "Author.csv")
PAPER_FILE = os.path.join(DATASET_PATH, "Paper.csv")
JOURNAL_FILE = os.path.join(DATASET_PATH, "Journal.csv")
CONFERENCE_FILE = os.path.join(DATASET_PATH, "Conference.csv")