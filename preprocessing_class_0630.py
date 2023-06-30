#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
import os
import shutil 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


# ### 1. class 제한 : 
# ### 2. BB 사이즈 제한 
# ### 3. annotation 변경

# In[2]:


Commen_Path = '/home/user/lmy/Traffic'
Path_list =['Seoul', 'Sejong', 'Daejeon_1', 'Daejeon_2']


# In[3]:


# Threshold_Class = '/home/user/1lmy/project/Dataset/Threshold_Class'
labels = '/home/user/lmy/Dataset/train/labels'
images = '/home/user/lmy/Dataset/train/images'


# ### class index annotation 설정

# In[6]:


traffic_class = [1400, 1401, 1300, 1301, 1405, 1501]
#change_class = [four_Blue, four_Red, three_Blue, three_Red, five_Red, five_Blue]
def change_class_str(category):
    if category == 1400:
        return 0
    elif category == 1401:
        return 1
    elif category == 1300:
        return 2
    elif category == 1301:
        return 3
    elif category == 1405:
        return 4
    elif category == 1501:
        return 5


# ### BoundingBox yolo 양식으로 변경

# In[7]:


# 이미지 사이즈 : 1536*2048 = 3145728
def change_BB(BB): # normalize 0~1
    X = ((BB[0] + BB[2])/2) /2048 # 너비  # (left + right) / 2
    Y = ((BB[1] + BB[3])/2) /1536 # 높이 (top + bottom) / 2
    W = (BB[2] - BB[0]) /2048 #너비 # right - left
    H = (BB[3] - BB[1]) /1536 # bottom - top
    yolo_bb = [X, Y, W, H]
    return(yolo_bb)


# In[8]:


class_coordinate, no_detect, size_che = [], [], []
Threshold = 1000 # 비율

for area in Path_list: # 지역 반복
    file_list = os.listdir(Commen_Path+"/"+area)
    for i in file_list: # 지역 내 반복
        if i.split(".")[1] == "txt":
            
            txt_file_path = os.path.join(Commen_Path, area, i)  #print(txt_file_path)            
            image_file_path = txt_file_path.split(".")[0]+'.jpg' #print(image_file_path)

            with open(txt_file_path, 'r') as file:
                contents = file.read()  # 파일 내용 읽기
                coordinate = list(map(int,contents.split())) # class, BoundingBox list
                if not coordinate:
                    no_detect.append(coordinate)
                elif coordinate[4] in traffic_class: # 클래스 조건
                    #class_coordinate.append(coordinate[4])
                    
                    img = cv2.imread(image_file_path) 
                    image_size = (list(img.shape)[0]*list(img.shape)[1]) 
                    bounging_size = (coordinate[2]-coordinate[0]) * (coordinate[3]-coordinate[1])
                    rate = int(3145728/bounging_size)
                    
                    if rate < Threshold:
                        size_che.append(image_size)
                        class_coordinate.append(coordinate[4])
                        
                        yolo_bb = change_BB(coordinate)
                        bbox_string = " ".join([str(x) for x in yolo_bb])
                        new_annotation = []
                        #yolo_class = change_class_str(coordinate[4])
                        new_annotation.append(f"{change_class_str(coordinate[4])} {bbox_string}")
                        #print(i)
                        if change_class_str(coordinate[4]) == None:
                            pass
                        else:
                            # new_annotation 은 file 생성하여 label 폴더에 저장
                            with open(os.path.join(labels, i), "w", encoding="utf-8") as f: # 세번째인자에 encoding
                                f.write("\n".join(new_annotation))
                            #shutil.copy(txt_file_path, labels)
                            # 이미지 파일은 복사해서 images 폴더에 저장
                            shutil.copy(image_file_path, images)


# In[9]:


len(no_detect)


# In[11]:


labels = '/home/user/lmy/Dataset/train/labels'
n_test = os.listdir(labels)
none_test = []
for i in n_test:
    txt_file_path = os.path.join(labels, i)
    with open(txt_file_path, 'r') as file:
        contents = file.read()
        coordinate = list(map(float,contents.split()))
        if len(coordinate) != 5:
            print("error")
        #else:
            #print('right')


# In[12]:


contents


# In[13]:


print(len(class_coordinate))
print(len(size_che))


# class가 존재하는 이미지 7997
# 
# bbox가 기준을 넘는 이미지 3633

# ### 클래스 비율 시각화

# In[14]:


df = pd.DataFrame(class_coordinate, columns =['category'])
df['category'].value_counts()


# In[ ]:


df2 = pd.DataFrame(size_che, columns =['category'])
df2['category'].value_counts()


# In[ ]:




