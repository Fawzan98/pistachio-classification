import streamlit as st
import pandas as pd
import tarfile
import gdown
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


st.sidebar.markdown('''
# Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data-preprocessing](#data-preprocessing)
- [Model Settings](#model-settings)
- [Model Evaluation](#model-evaluation)

''', unsafe_allow_html=True)


st.title('Pistachio Classification Prediction App')
st.header('Introduction')
st.write("""
This app capable on making **classification** on Pistachio images by feeding the various types of Pistachio images into a deep learning model named VGG116.\n
VGG16 is a convolutional neural network with 16 layers deep and the model used transfer learning where the model has been pre-trained through more than a million images from ImageNet database.\n
Coding source inspiration : [Pistachio Detection by Ayush Verma](https://www.kaggle.com/code/ayushv322/pistachio-detection-vgg16-97-9) 
and [Step By Step VGG16 Implementation In Keras For Beginners](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c)
""")

from PIL import Image
img1 = Image.open('VGG16.png')
st.image(img1, caption = "VGG16 network architecture", use_column_width='always')

st.header('Dataset')
st.write('''
The datasets were obtained in Kaggle which contain a total of 2148 images, 1232 of Kirmizi and 916 of Siirt Pistachio.\n
''')

img2 = ['kirmizi (23).jpg', 'siirt 24.jpg']
st.image(img2,width=300, caption=["Kirmizi Pistachio", "Siirt Pistachio"])

st.write('''
We can see that both Pistachios' look very similar though some of them has distinct features and sometimes can be difficult to recognize from naked eye.
Now we will be using an Artificial Intelligence to make it learn to make predictions between Kirmizi and Siirt Pistachio.
The datasets were split into two: training sets and testing sets which contain 80% and 20% respectively.
''')

st.header('Data-preprocessing')
st.subheader('Image Augmentation')

st.write('''
A a technique of applying different transformations to original images which results in multiple transformed copies of the same image.
This technique not only expand the size of your dataset but also incorporate a level of variation in the dataset which allows your model to generalize better on unseen data. 
Also, the model becomes more robust when it is trained on new, slightly altered images.
''')

img3 = Image.open('augment.png')
st.image(img3, caption = "Example of image augmentation", use_column_width='always')


st.header('Model Settings')
st.write('''
The hyperparameter used in this model shown as listed below:
''')

hyperparameter = {
  "Settings": ["Input Shape", "Epoch", "Learning Rate", "Dropout", "Activation", "Optimizer", "Early Stopping"],
  "Descriptions": ["512 x 512", "50", "0.005", "0.1", "Softmax", "Adam", "10 Validation Loss"]
}

df = pd.DataFrame(hyperparameter)
st.table(df)

img4 = Image.open('modeltrain.JPG')
st.image(img4, caption = "Model in training", use_column_width='always')


path = "content/model/vgg16_1.h5"
if not os.path.exists(path):
  url = "https://drive.google.com/file/d/1gqzmrh1SAzVQdKM0usFnDL_eGkJJcJht/view?usp=sharing"
  output = "model.tar.gz"
  gdown.download(url=url, output=output, quiet=False, fuzzy=True)
  file = tarfile.open("model.tar.gz")
  file.extractall()
  file.close()

#st.write(os.listdir())


model = load_model('./content/model/vgg16_1.h5')
  
st.header('Model Evaluation')
st.write('''
After the training stopped at epoch 39 via early stopping, it is time to evaluate the model.
''')





