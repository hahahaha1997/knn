#encoding=utf-8
from numpy import *
import operator

def createdataset(filename):
    with open(filename,'r')as csvfile:
        dataset = [line.strip().split(',')for line in csvfile.readlines()]
        del(dataset[-1])
        cleanoutdata(dataset)#数据清洗
        dataset=precondition(dataset)#数据预处理
        labels=[each[-1]for each in dataset]#获取最后一列数据，即类别
        dataset=[each[0:-2]for each in dataset]#去除最后一列数据，即类别
        return array(dataset),labels

def cleanoutdata(dataset):#数据清洗
    for i in range(5):
        for row in dataset:
            for column in row:
                if column == '?'or column=='':
                    dataset.remove(row)
                    break

def precondition(dataset):
    dict={'Private':0,'Self-emp-not-inc':1,'Self-emp-inc':2,'Federal-gov':3,
          'Local-gov':4,'State-gov':5,'Without-pay':6,'Never-worked':7,
          'Bachelors':0,'Some-college':1,'11th':2,'HS-grad':3,'Prof-school':4,
          'Assoc-acdm':5,'Assoc-voc':6,'9th':7,'7th-8th':8,'12th':9,'Masters':10,
          '1st-4th':11,'10th':12,'Doctorate':13,'5th-6th':14,'Preschool':15,
          'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2,
          'Separated':3, 'Widowed':4, 'Married-spouse-absent':5,
          'Married-AF-spouse':6,'Tech-support':0, 'Craft-repair':1,
          'Other-service':2, 'Sales':3, 'Exec-managerial':4, 'Prof-specialty':5,
          'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Adm-clerical':8,
          'Farming-fishing':9, 'Transport-moving':10, 'Priv-house-serv':11,
          'Protective-serv':12, 'Armed-Forces':13,'Wife':0, 'Own-child':1,
          'Husband':2, 'Not-in-family':3, 'Other-relative':4, 'Unmarried':5,
          'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2, 'Other':3,
          'Black':4,'Female':0,'Male':1,'United-States':0, 'Cambodia':1,
          'England':2, 'Puerto-Rico':3, 'Canada':4, 'Germany':5,
          'Outlying-US(Guam-USVI-etc)':6, 'India':7, 'Japan':8, 'Greece':9,
          'South':10, 'China':11, 'Cuba':12, 'Iran':13, 'Honduras':14,
          'Philippines':15, 'Italy':16, 'Poland':17, 'Jamaica':18, 'Vietnam':19,
          'Mexico':20, 'Portugal':21, 'Ireland':22, 'France':23,
          'Dominican-Republic':24, 'Laos':25, 'Ecuador':26, 'Taiwan':27,
          'Haiti':28, 'Columbia':29, 'Hungary':30, 'Guatemala':31,
          'Nicaragua':32, 'Scotland':33, 'Thailand':34, 'Yugoslavia':35,
          'El-Salvador':36, 'Trinadad&Tobago':37, 'Peru':38, 'Hong':39,
          'Holand-Netherlands':40,'<=50K':'<=50K','>50K':'>50K','<=50K.':'<=50K','>50K.':'>50K','?':0}
    dataset = [[int(column.strip()) if column.strip().isdigit() else dict[column.strip()] for column in row] for row in dataset]#对于数据集中每一个元素，如果是离散性数据，转换为数值型
    return dataset

def norm(dataset):#归一化数据，将所有的数据集中在【0，1】中，保证取值比较大的数据对于距离的影响不会太大
    minvals=dataset.min(0)
    maxvals=dataset.max(0)
    ranges=maxvals-minvals
    normdata=zeros(shape(dataset))
    m=dataset.shape[0]
    normdata=dataset-tile(minvals,(m,1))
    normdata=normdata/tile(ranges,(m,1))
    return normdata


def classify(testdataset,dataset,testlabels,labels,k,correct):
    datasetsize=dataset.shape[0]
    j=0
    for vec in testdataset:
        diff=tile(vec,(datasetsize,1))-dataset  #计算距离，直接计算矩阵距离，不需要按列计算距离
        sqDiffMat = diff ** 2;
        sqDistances = sqDiffMat.sum(axis=1)
        distance = sqrt(abs(sqDistances))
        sorteddistane=distance.argsort()        #将距离矩阵进行排序
        count={}#保存前k个距离当前向量最近的点的类别
        for i in range(k):#对于前k个
            label=labels[sorteddistane[i]]                  #在测试集中属于什么类别
            count[label]=count.get(label,0)+1               #计算每种类别的个数
        sortedcount=sorted(dict2list(count),key=operator.itemgetter(1),reverse=True)        #排序计算各种类别的个数
        if sortedcount[0][0]==testlabels[j]:        #如果最多的类别和测试集原本的类别相同，则增加一个correct表示分类正确
            correct=correct+1
        j+=1
    return correct

def dict2list(dic:dict):#将字典转换为list类型
    keys=dic.keys()
    values=dic.values()
    lst=[(key,value)for key,value in zip(keys,values)]
    return lst

correct=0
datasetname = r"C:\Users\yang\Desktop\adult.data"           #训练集
dataset,labels=createdataset(datasetname)

testdatasetname=r"C:\Users\yang\Desktop\adult.test"         #测试集
testdataset,testlabels=createdataset(testdatasetname)

k=1
correct=classify(testdataset,dataset,testlabels,labels,k,correct)  #分类过程
print("准确率：")
print(correct/len(testdataset))
correct=0
k=3
correct=classify(testdataset,dataset,testlabels,labels,k,correct)  #分类过程
print("准确率：")
print(correct/len(testdataset))