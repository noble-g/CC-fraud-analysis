from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from numerize.numerize import numerize
import plotly.express as px
import plotly.figure_factory as ff
import os

## import Data
current_dir = os.path.dirname(__file__)
relative_path = os.path.join(current_dir, "materials", "creditcard.csv")
# Load the CSV file using the relative path
df = pd.read_csv(relative_path)

#file = "materials/creditcard.csv"
#df = pd.read_csv(file)
# Configure streamlit page
st.set_page_config(page_title = "CC Fraud", page_icon = "🚩",layout="wide")

# Sidebar
st.sidebar.header("Side Bar")
#model = st.sidebar.select("Choose model", )

# Title
st.header('Credit card Fraud Detection Analysis Using ML and DL Models')

# Cards

col1, col2, col3, col4 = st.columns(4)
col1.metric("Number of rows", df.shape[0])
col2.metric("Number of columns", df.shape[1])
col3.metric("Number of Fraudulent Transactions", df['Class'].sum())
col4.metric("Number of Normal Transactions", df.shape[0] - df['Class'].sum())

# Table
st.subheader('Dataset')
st.write(df)

corr = Image.open(os.path.join(current_dir, "materials", "correlation_matrix.png"))
scat_mat = Image.open(os.path.join(current_dir, "materials", "scatterplot_matrix.png"))

chart1, chart2 = st.columns(2)
# Correlation Matrix
with chart1:
    st.image(corr, caption="Correlation", use_column_width=True)
    
# Scatterplot
with chart2:
    st.image(scat_mat, caption="Scatterplot Matrix", use_column_width=True)

# Explaining SMOTE
st.subheader('SMOTE - Synthetic Minority Oversampling Technique')
st.write("Synthetic Minority Over-sampling Technique (SMOTE), is a method designed to address the issue of class imbalance in machine learning datasets. In many real-world classification problems, the number of examples belonging to one class significantly outweighs the number of examples in another class. This imbalance can lead to biased models that favor the majority class and perform poorly on minority class predictions. SMOTE tackles this problem by generating synthetic examples for the minority class. It does so by identifying minority class instances and creating new synthetic instances along the line segments joining them. By strategically synthesizing new data points, SMOTE effectively rebalances the class distribution, providing more robust training data for machine learning models. This technique helps to alleviate the bias towards the majority class and improves the model's ability to accurately classify instances from the minority class. However, it's essential to use SMOTE judiciously and consider potential drawbacks, such as the introduction of synthetic noise or overfitting, in order to achieve optimal results.")
# Before and after SMOTE
b4 = Image.open(os.path.join(current_dir, "materials", "b4_SMOTE_pie_plot.png"))
afta = Image.open(os.path.join(current_dir, "materials", "afta_SMOTE_pie_plot.png"))

before, after = st.columns(2)
with before:
    st.image(b4, caption="Before SMOTE", use_column_width=True)
with after:
    st.image(afta, caption = "After SMOTE", use_column_width=True)

st.write("by Salaudeen Gbolahan")