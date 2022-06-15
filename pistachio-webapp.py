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
out_dir = "images/kirmizi"

data_url = "https://github.com/Fawzan98/pistachio-classification/tree/main/img_testing/"

#os.mkdir("images")
#i = 1
#while i <= 10:
#  wget.download(data_url + f"Kirmizi/kirmizi ({i}).jpg", out = "images")
 # wget.download(data_url + f"Siirt/siirt ({i}).jpg", out = "images")
 # i = i + 1
#os.chdir("images")
#os.remove("pistachio_imgdataset (1).rar")
#st.write(os.listdir())

st.header("Model")
st.write("""
Now we will try the accuracy of the model in predicting between Kirmizi and Siirt Pistachio.
""")

#shutil.unpack_archive("pistachio_imgdataset.rar","/content", "rar")
#pt.extract_archive("pistachio_imgdataset.rar", outdir='/content')
#rar = rarfile.RarFile('pistachio_imgdataset.rar')







