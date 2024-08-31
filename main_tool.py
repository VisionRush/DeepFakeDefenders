import os
import numpy as np
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.dsproc_mclsmfolder import MultiClassificationProcessor_mfolder


###############################################
# 生成训练数据列表
###############################################
def generateTrainingDataList(dataset_list, img_all_txtPath, label_txtPath):
    dataset = MultiClassificationProcessor_mfolder()
    dataset.load_data_from_dir(dataset_list)

    # write image list txt
    file = open(img_all_txtPath, 'w')
    
    for i in range(len(dataset.img_names_)):
        img_path = dataset.img_paths_[i]
        img_label = dataset.img_labels_[i]
        
        file.write(img_path + ' ' + str(img_label) + '\n')
        
    # write category list txt
    file = open(label_txtPath,'w')
    for key, value in dataset.ctg_name2idx_.items():
        file.write(key + ' ' + str(value) + '\n')


###############################################
# 生成训练数据列表 + 随机划分数据集
###############################################
def randomPartitionTheDataset(img_all_txtPath, label_txtPath, trainset_file_txtPath, valset_file_txtPath):
    
    dataset = MultiClassificationProcessor_mfolder()
    dataset.load_data_from_txt(img_all_txtPath, label_txtPath)

    # split
    idxs = np.arange(len(dataset.img_names_))
    random.shuffle(idxs)
    trainset_pos = int(len(idxs) * 1)
    
    # write image list txt
    trainset_file = open(trainset_file_txtPath, 'w')
    valset_file = open(valset_file_txtPath, 'w')
    
    for i in range(len(dataset.img_names_)):
        img_path = dataset.img_paths_[idxs[i]]
        img_label = dataset.img_labels_[idxs[i]]
        if i < trainset_pos:
            trainset_file.write(img_path + ' ' + str(img_label) + '\n')
        else:
            valset_file.write(img_path + ' ' + str(img_label) + '\n')

   
# 多分类训练标签
def faceImg_trainDatasets_process():
    dataset_list ={
        "real":   [''],
        "fake":   [''],
    }

    train_data_path = "/data/all_data"
    
    os.makedirs(train_data_path, exist_ok=True)
    
    # 全部图像数据路径
    img_all_txtPath =  train_data_path + '/faceImg_all.txt'
    # 训练的数据标签
    label_txtPath = train_data_path + '/faceImg_label.txt'
    
    # 生成训练数据及标签
    generateTrainingDataList(dataset_list, img_all_txtPath, label_txtPath)
    
    trainset_file_txtPath = train_data_path + '/faceImg_list_train.txt'
    valset_file_txtPath = train_data_path + '/faceImg_list_val.txt'
    
    # 随机划分数据集：训练集、测试集、验证集
    randomPartitionTheDataset(img_all_txtPath, label_txtPath, trainset_file_txtPath, valset_file_txtPath)
    

if __name__ == '__main__':
    faceImg_trainDatasets_process()
