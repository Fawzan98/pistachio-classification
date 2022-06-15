import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import splitfolders
import streamlit as st
import wget
import patoolib as pt
import random
import tarfile
import urllib.request
import gdown
from pathlib import Path

st.write("""
# Pistachio Classification Prediction App
This app capable on making **classification** on Pistachio images by feeding the various types of Pistachio images into a deep learning model named VGG116.\n
VGG16 is a convolutional neural network with 16 layers deep and the model used transfer learning where the model has been pre-trained through more than a million images from ImageNet database.\n
Coding source inspiration : [Pistachio Detection by Ayush Verma](https://www.kaggle.com/code/ayushv322/pistachio-detection-vgg16-97-9) 
and [Step By Step VGG16 Implementation In Keras For Beginners](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c)
""")

st.header("Dataset")

st.write("""
The datasets were obtained in Kaggle which contain a total of 2148 images, 1232 of Kirmizi and 916 of Siirt Pistachio.\n
The datasets then split into 80% for training and 20% for testing phase.
""")

data_url = "https://github.com/Fawzan98/pistachio-classification/tree/main/img_testing/"

#os.remove('images')

fle=Path('images')

if fle.is_file():
  st.write("File exist! Displaying random image of Kirmizi and Siirt Pistachio.")
else:
  os.mkdir("images")
  i = 1
  while i <= 10:
    wget.download(data_url + f"Kirmizi/kirmizi ({i}).jpg", out = "images/Kirmizi_Pistachio")
    wget.download(data_url + f"Siirt/siirt ({i}).jpg", out = "images/Siirt_Pistachio")
    i = i + 1

labels = ['Siirt_Pistachio', 'Kirmizi_Pistachio']
data_dir = "/images"
    
def load_random_imgs_from_folder(folder,label):
  plt.figure(figsize=(15,15))
  for i in range(3):
    file = random.choice(os.listdir(folder))
    image_path = os.path.join(folder, file)
    img=mpimg.imread(image_path)
    ax=plt.subplot(1,3,i+1)
    ax.title.set_text(label)
    plt.xlabel(f'Name: {file}')
    plt.imshow(img)
for label in labels:
    load_random_imgs_from_folder(f"{data_dir}/{label}",label)


  
  
  
#os.chdir("images")


st.header("Model")
st.write("""
Now we will try the accuracy of the model in predicting between Kirmizi and Siirt Pistachio.
""")

#url = "https://drive.google.com/file/d/1gqzmrh1SAzVQdKM0usFnDL_eGkJJcJht/view?usp=sharing"
#output = "model.tar.gz"
#gdown.download(url=url, output=output, quiet=False, fuzzy=True)

#file = tarfile.open("model.tar.gz")
#file.extractall()
#file.close()

st.write(os.listdir())







