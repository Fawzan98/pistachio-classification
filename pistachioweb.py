import streamlit as st
import pandas as pd
import tarfile
import gdown
import os
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
#import preprocess


st.sidebar.markdown('''
# Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data-preprocessing](#data-preprocessing)
- [Model Settings](#model-settings)
- [Model Evaluation](#model-evaluation)
- [Model Application](#model-application)

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
We can see that both Pistachios' look very similar through the naked eye and sometimes they may be difficult to differentiate. 
Thus to solve the problem we will be using an Artificial Intelligence to make predictions between Kirmizi and Siirt Pistachio.
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
  
  
st.header('Model Evaluation')
st.write('''
After the training stopped at epoch 39 via early stopping, it is time to evaluate the model. 
''')
img5 = Image.open('validation loss.JPG')
st.image(img5, caption = "Model validation loss graph", use_column_width='always')

st.write('''
Throughout 39 epochs, we can see that initially the loss is high for both training and validation test, and
later they gradually decrease as the model continuosly improve its weights and bias to achieve better results.
''')

img6 = Image.open('accuracy.JPG')
st.image(img6, caption = "Model accuracy graph", use_column_width='always')

st.write('''
In this graph, we can see the trends in training and validation gradually increase. This is good because the model
is indeed learning and trying to become better at every epochs. We do not want the model to become overfit which
affecting the trend of validation test to decrease. Thus, early stopping method is great to keep the model in best
performance.
''')

st.header('Model application')

class_dict = {'Kirmizi Pistachio': 0,
              'Siirt Pistachio': 1
}

class_names = list(class_dict.keys())
saved_model = load_model('./content/model/vgg16_1.h5')

img_path = "img/testing"

def choose_files():
  img_path = "img_testing"
  random_folder = random.choice(os.listdir(img_path))
  random_pic = random.choice(os.listdir(img_path+"/"+random_folder))
  full_path = img_path + "/" + random_folder + "/" + random_pic
  img8 = Image.open(full_path)
  st.image(img8, caption = full_path, width = 200)
  return full_path

x,y,z = choose_files()  
#img7 = Image.open(img_path)
#st.image(img7, caption = img_path, width = 200)

#img_path = "img_testing/Kirmizi/kirmizi (3).jpg"
play = st.radio(
  "Choose your Pistachio:", 
  (x, y, z)
)


#img_path = "img_testing/Siirt/siirt (3).jpg"


if play == x:
  st.write('You have choose x')
elif play == y:
  st.write('You have choose y')
else:
  st.write('You have choose z')

#img_size = 512
#imges = cv2.imread(img_path)
#test_image = cv2.resize(imges, (int(img_size), int(img_size)))
#test_image = np.array(test_image)
#test_image = np.expand_dims(test_image, axis=0)
#img_class = saved_model.predict(test_image)
#img_class = img_class.flatten()
#m = max(img_class)
#for index, item in enumerate(img_class):
#    if item == m:
#        pred_class = class_names[index]
#st.write(pred_class)



#test_image = preprocess(test_image)
#test_image = edge_and_cut(test_image)





