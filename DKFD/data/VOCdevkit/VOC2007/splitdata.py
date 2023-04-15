import os
import random

xmlfilepath = r'data/VOCdevkit/VOC2007/Annotations'

txtsavepath = r'data/VOCdevkit/VOC2007/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)
random.shuffle(total_xml)
print(len(total_xml))

#提取原始测试集的所有图像名称
with open(r'C:data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r') as f:
    testlist=f.read().splitlines()#读取每一行不带\n
print(len(testlist))
trainval=[]
for item in total_xml:
    orgimageName=item.replace('.xml','')
    if orgimageName not in testlist:
        trainval.append(item.replace('.xml',''))
#print(len(testlist))
#print(len(trainval))
print(len(trainval))
with open(r'/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w') as ftrainval:
    for item in trainval:
        ftrainval.write(item+'\n')
