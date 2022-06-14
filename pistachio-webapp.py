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
This app will predicts the **Pistachio** type by using VGG16, a convolutional neural network with 16 layers deep!
Furthermore, the model used transfer learning where the model has been pre-trained from ImageNet database.
""")

st.title("The app title")
st.header("Pistachio Classification Prediction App")
