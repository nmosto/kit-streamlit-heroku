import joblib
import os
import io
import sys

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import cv2

import collections
from collections import defaultdict
import operator
from skimage.io import imread, imshow
from skimage import feature
from PIL import Image as IMG
from scipy.stats import itemfreq

# Import Functions for Image Features
from brightness_dullness import color_analysis
from brightness_dullness import perform_color_analysis
from uniformity import uniformity
from dominant_color_clustering import dominant_color
from average_color import average_color
from blurriness import blurriness
from img_size import compression_size
from img_size import get_dimensions

st.set_option('deprecation.showfileUploaderEncoding', False)

#B eginning of app text
st.markdown("<h1 style='text-align: center; color: black;'>Kick-Into-Traction</h1>", unsafe_allow_html=True)
st.write('### First impressions make a difference.')
st.write('')
st.write('### Crowdfunding campaigns can fail due to using poor images.')
st.write('')
st.write('Kick-Into-Traction is here to help your campaign get momentum quickly by focusing on what people see first: **The Cover Photo**')
st.write('')
st.write('### Have your cover photo analyzed for red flags.')

# Image uploader
uploaded_file = st.file_uploader("Submit Your Campaign Cover Image Here:")


# Read in saved logistic regression model

#load_logreg = pickle.load(open('kit_logreg.pkl', 'rb'))



# Code to execute once file is uploaded
if uploaded_file is not None:

    image = IMG.open(uploaded_file)
    saved_img = image.save('saved_img.jpeg')
    st.image(image, caption='Uploading Successful.', use_column_width=True)
    st.write("")
    #st.write("Analyzing...")

    # Do calculations
    dullness = perform_color_analysis(uploaded_file, 'black')
    brightness = perform_color_analysis(uploaded_file, 'white')
    uniformity = uniformity(uploaded_file)
    blurriness = blurriness('saved_img.jpeg')
    compression_size = compression_size('saved_img.jpeg')


    dc = dominant_color('saved_img.jpeg')
    # Normalize RGB values from 256 to 1
    dom_red = np.round(dc[0]/255,2)
    dom_green = np.round(dc[1]/255,2)
    dom_blue = np.round(dc[2]/255,2)
    # Normalize RGB values from 256 to 1
    ac = average_color('saved_img.jpeg')
    ave_red = np.round(ac[0]/255,2)
    ave_green = np.round(ac[1]/255,2)
    ave_blue = np.round(ac[2]/255,2)



    #print(x)

    # Populate DataFrame
    data1 = {
            'compression_size': compression_size,
                                    }


    data2 = {
            'dullness': dullness,
            'brightness': brightness,
            'uniformity': uniformity,
            'blurriness': blurriness,
                                    }

    data3 = {
            'dom red' : dom_red,
            'dom green' : dom_green,
            'dom blue' : dom_blue,
            'ave red' : ave_red,
            'ave green' : ave_green,
            'ave blue' : ave_blue
                                    }

    # Display the image data in three dataframe charts
    st.write('## Your Image Features')
    st.write('**Compression (bytes)**')
    features1 = pd.DataFrame(data1, index=[0])
    st.dataframe(features1)
    st.write('**Structural Features (a.u.)**')
    features2 = pd.DataFrame(data2, index=[0])
    st.dataframe(features2)
    st.write('**Colors (0-1)**')
    features3 = pd.DataFrame(data3, index=[0])
    st.dataframe(features3)

    #
    keys = ['dullness','brightness','uniformity','average_red','average_green',
                     'average_blue','dominant_red','dominant_green','dominant_blue','blurriness',
                     'compression_size']

    values = np.array([dullness, brightness, uniformity, ave_red, ave_green, ave_blue,
                             dom_red, dom_green, dom_blue, blurriness,
                             compression_size])

    dictionary = dict(zip(keys,values))



    # Import model
    modelname = 'kit_RFM_10_20_20.pkl'
    model = joblib.load(modelname)
    inp_data = values.reshape(1,-1)
    pred = model.predict(inp_data)
    st.write('')
    if pred == 0:

        st.write('### Our machine learning models predict that your campaign cover image will **NOT** make a good impression.')
    else:
        st.write('### Our machine learning models predict that your campaign cover image is sufficiently high in quality to make a good impression, but there is always room for improvement.')



    st.write('')
    st.write('## What can you do to improve?')
    st.write('1.) The **2** most significant features are: Compression Size and Uniformity.')
    st.write('2.) Look at how past successful campgain images stack up in comparison to yours!')
    st.write('3.) If your image scores differently, consider post-processing improvements to augment the key features.')
    # Suggestions
    st.write('## Suggesions for Improving your Image')
    st.write('')
    #log_reg = 'logreg_10_2_20.pkl'
    #pipe = joblib.load(log_reg)





    # Import training data for recommnedation figures
    features_table = 'final_features_df1.pkl'
    df = joblib.load(features_table)

    @st.cache
    def load_data():
        data = joblib.load(features_table)
        return data

    df = load_data()
    # Separate the funded and unfunded projects
    funded_projects = df[df['state'] == 1]
    #failed_projects = df[df['state'] == 0]



    # Figures for suggestions

    # Compression Size
    st.write('# Compression Size')
    fig, ax = plt.subplots()
    ax.hist(df['compression_size']*1e-6,bins = 1000)
    l2 = ax.axvline(x=compression_size*1e-6, color='red', label='Your Image: '+str(np.round(compression_size*1e-6, 4))+' MB')
    ax.set_title('Successful Campaigns Histogram - Compression Size')
    ax.set_xlim([0,2])
    ax.axes.yaxis.set_ticks([])
    ax.legend(handles=[l2])
    ax.set_xlabel('Compression Size (MB)')
    ax.set_ylabel('Relative Frequency')
    st.pyplot(fig)
    st.write('### **If compression is very low, your image is likely of poor quality.**')
    st.write('')

    # Uniformity
    st.write('# Uniformity')
    fig, ax = plt.subplots()
    ax.hist(df['uniformity'],bins = 500)
    l2 = ax.axvline(x=uniformity, color='red', label='Your Image: '+str(np.round(uniformity, 4))+' a.u.')
    ax.set_title('Successful Campaigns Histogram - Uniformity')
    ax.axes.yaxis.set_ticks([])
    ax.legend(handles=[l2])
    ax.set_xlabel('Uniformity (a.u.)')
    ax.set_ylabel('Relative Frequency')
    st.pyplot(fig)
    st.write('### **If uniformity is very low, your image shows little pixel variation, has few edges and maybe uninteresting.**')
    st.write('')





    # Remove file created for openCV processing
    os.remove('saved_img.jpeg')
