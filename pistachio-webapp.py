import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import splitfolders
import streamlit as st

st.write("""
# Pistachio Classification Prediction App
This app capable on making **classification** on Pistachio by feeding the various types of Pistachio into a deep learning model named VGG116.\n
VGG16 is a convolutional neural network with 16 layers deep and the model used transfer learning where the model has been pre-trained through more than a million images from ImageNet database.\n
Coding source inspiration : [Pistachio Detection by Ayush Verma](https://www.kaggle.com/code/ayushv322/pistachio-detection-vgg16-97-9) 
and [Step By Step VGG16 Implementation In Keras For Beginners](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c)
""")

st.title("The app title")
st.header("Pistachio Classification Prediction App")
