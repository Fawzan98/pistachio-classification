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
This app will predicts the **Pistachio** based on its type by using VGG16, a convolutional neural network with 16 layers deep!\n
Furthermore, the model used transfer learning where the model has been pre-trained from ImageNet database.\n
Coding source inspiration : [Pistachio detection by Ayush Verma](https://www.kaggle.com/code/ayushv322/pistachio-detection-vgg16-97-9) 
and [Step by step VGG16 implementation in Keras for beginners](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c)
""")

st.title("The app title")
st.header("Pistachio Classification Prediction App")
