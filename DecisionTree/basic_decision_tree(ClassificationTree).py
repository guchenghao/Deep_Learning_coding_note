import matplotlib.pyplot as plt
import math
import operator
import pickle

def createDataSet():
    dataSet = [
        [0, 0, 0, 0, "no"],
        [0, 0, 0, 1, "no"],
        [0, 1, 0, 1, "yes"],
        [0, 1, 1, 0, "yes"],
        [0, 0, 0, 0, "no"],
        [1, 0, 0, 0, "no"],
        [1, 0, 0, 1, "no"],
        [1, 1, 1, 1, "yes"],
        [1, 0, 1, 2, "yes"],
        [1, 0, 1, 2, "yes"],
        [2, 0, 1, 2, "yes"],
        [2, 0, 1, 1, "yes"],
        [2, 1, 0, 1, "yes"],
        [2, 1, 0, 2, "yes"],
        [2, 0, 0, 0, "no"]
    ]

    labels = ["F1-AGE", "F2-WORK", "F3-HOME", "F4-LOAN"]
    
    return dataSet, labels


# * featLabels数组是用于存储最佳feature选择的顺序
def createTree(dataset, labels, featLabels):
    # * 取出类别标签
    classList = [example[-1] for example in dataset]
    
    # * 如果当前节点中都属于一个类别, 停止分裂
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    if len(dataset[0]) == 1:
        return majorityCnt(classList)  # * 计算当前节点中，哪个类别最多，众数

    bestFeature = chooseBestFeatureToSplit(dataset)  # * bestFeature是一个索引值
    bestFeatureLabel = labels[bestFeature]
    featLabels.append(bestFeatureLabel)

    myTree = {bestFeatureLabel: {}}  # * 双重字典

    del labels[bestFeature]  # * 选择了这个feature，将这个feature删除
    
    
    # * 统计当前选择的feature中属性值有几种
    featureValues = [example[bestFeature] for example in dataset]
    unique_values = set(featureValues)
    
    
    
    for value in unique_values:
        sublabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataset, bestFeature, value), sublabels, featLabels)
    
    
    return myTree


# * vote
def majorityCnt(classList):
    classCount = {}
    
    for vote in classList:
        if vote in classList:
            classCount[vote] += 1
        
        else:
            classCount[vote] = 1
            
    class_count_sorted  = sorted(classCount.items(), key=lambda x: x[1])
    
    return class_count_sorted[-1][0]


# * 核心逻辑
# * 选择特征节点
def chooseBestFeatureToSplit(dataset):

    numFeatures = len(dataset[0]) - 1

    baseEntropy = calShannonEnt(dataset)

    bestInfoGain = 0

    bestFeature = -1  # * 索引值

    for i in range(numFeatures):
        featList = [sample[i] for sample in dataset]

        unique_values = set(featList)

        valueEntropy = 0
        for value in unique_values:

            subDataSet = splitDataSet(dataset, i, value)

            ratio_weighted = float(len(subDataSet) / len(dataset))   # * 当前取值的样本占上一个节点样本总数的比例

            valueEntropy += ratio_weighted * calShannonEnt(subDataSet)
        
        
        
        infoGain = baseEntropy - valueEntropy
        
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
            
    
    return bestFeature


def splitDataSet(dataset, index_feature, value):
    retDataSet = []
    
    for sample in dataset:
        if sample[index_feature] == value:
            reducedFeature = sample[:index_feature]
            reducedFeature += sample[index_feature + 1:]
        
            retDataSet.append(reducedFeature)
    
    
    return retDataSet


# * 计算熵值
# * 计算每个不同类别的子集占样本总量概率，以及信息熵
def calShannonEnt(dataset):
    num_samples = len(dataset)

    labelCount = {}

    for sample in dataset:
        current_label = sample[-1]

        if current_label in labelCount:
            labelCount[current_label] += 1

        else:
            labelCount[current_label] = 1

    shannonEnt = 0

    for label, count in labelCount.items():

        prop = float(count / num_samples)

        cur_weighted_information = - prop * math.log(prop, 2)

        shannonEnt += cur_weighted_information
        
    
    
    return shannonEnt







if __name__ == "__main__":
    dataset, labels = createDataSet()
    featureLabels = []
    myTree = createTree(dataset, labels, [featureLabels])
    
    