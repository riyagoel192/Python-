#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


img = plt.imread("demo.jpg")
plt.imshow(img)


# In[4]:


template= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faces=template.detectMultiScale(img,1.1,5)
faces


# In[5]:


faces.shape[0]


# In[6]:


for face in faces:
        x,y,w,h = face
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0),2)
plt.imshow(img)


# In[3]:


cam = cv2.VideoCapture(0)
#cam.release()


# In[ ]:


while True:
    ret, frame = cam.read()   # cam.read() returns 2 things: 1 boolean value if it is able to read image and 1 frame value
    
    template= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    if ret==False:
        continue
        
    faces= template.detectMultiScale(frame,1.1,5)
    
    for face in faces:
        x,y,w,h= face
        img= cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0),2)
        
    key_pressed = cv2.waitKey(1) &0xFF
    
    if(key_pressed==ord('q')):
        break;
    cv2.imshow("frame",frame)
cam.release()
cv2.destroyAllWindows()


# In[ ]:




