#!/usr/bin/env python
# coding: utf-8

# In[17]:


from glob import glob
import os
import shutil 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[18]:


import sys
print(sys.version)


# In[21]:


images = os.listdir('/home/user/lmy/Dataset/train/images')
file_name = []
for i in images:
    if i.startswith('.'):
        continue
    file_name.append(i)
file_name = sorted(file_name)
file_name = np.array(file_name)
val_split = np.random.choice(len(file_name),600)
val_file = file_name[val_split]


# In[22]:


print(len(file_name), len(val_file))
val_file


# In[23]:



# 전체 있던 곳
folder_img = "/home/user/lmy/Dataset/train/images/"
folder_label = "/home/user/lmy/Dataset/train/labels/"
# 이동할 곳 
val_folder_img = "/home/user/lmy/Dataset/valid/images/"
val_folder_label = "/home/user/lmy/Dataset/valid/labels/"
for i in val_file:
    name = i.split(".")[0]
    try:
        shutil.move(folder_img + name + ".jpg", val_folder_img)
        shutil.move(folder_label + name + ".txt", val_folder_label)
    except:
        continue


# In[24]:


print(len(os.listdir(val_folder_img)))
print(len(os.listdir(val_folder_label)))


# In[25]:


get_ipython().system('pwd')


# In[ ]:


### 4. yaml 파일 생성


# In[26]:


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


# In[27]:


# yolov8
import yaml
data = {
    "train" : "/home/user/lmy/Dataset/train/images/",
    "val" : "/home/user/lmy/Dataset/valid/images/",
    
    "names" : ['four_Blue', 'four_Red', 'three_Blue', 'three_Red', 'five_Blue', 'five_Red'],
    "nc" : 6
}

with open("./data.yaml","w") as file:
    yaml.dump(data,file)

with open("./data.yaml","r") as file:
    Traffic_yaml = yaml.safe_load(file)
    display(Traffic_yaml)


# In[13]:


get_ipython().system('pwd')


# In[1]:


from ultralytics import YOLO


# In[2]:


# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.model.args['device'] = 1
model.train(data="/home/user/lmy/data.yaml", epochs=100, batch=4, seed=123, imgsz=(1280,1280),
            optimizer='Adam', lr0=1e-3)
metrics = model.val()  # evaluate model performance on the validation set


# In[29]:


get_ipython().system('nvidia-smi')


# In[20]:


get_ipython().system('pwd')

