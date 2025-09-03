import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Student Marks Predictor",
    layout="wide"
)

# Title and description
st.title(" Student Marks Prediction App")
st.markdown("""
This app predicts student marks based on:
- Hours studied
- Sleep hours
- Previous score
""")

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    hours_studied = st.sidebar.slider('Hours Studied', 1.0, 10.0, 4.0)
    sleep_hours = st.sidebar.slider('Sleep Hours', 4.0, 10.0, 6.0)
    previous_score = st.sidebar.slider('Previous Score', 0, 100, 50)
    return hours_studied, sleep_hours, previous_score

# Get user input
hours, sleep, prev_score = user_input_features()

# Sample data
data = pd.read_csv("student_scores.csv")
df = pd.DataFrame(data)

# Train the model
X = df[['Hours_Studied', 'Sleep_Hours', 'Previous_Score']]
y = df['Marks']

model = LinearRegression()
model.fit(X, y)

# Make prediction
prediction = model.predict([[hours, sleep, prev_score]])

# Display results
st.subheader('Prediction')
st.metric(label="Predicted Marks", value=f"{prediction[0]:.2f}")

# Show model details
st.subheader('Model Details')
col1, col2, col3, col4 = st.columns(4)
col1.metric("Intercept", f"{model.intercept_:.2f}")
col2.metric("Hours Coefficient", f"{model.coef_[0]:.2f}")
col3.metric("Sleep Coefficient", f"{model.coef_[1]:.2f}")
col4.metric("Previous Score Coefficient", f"{model.coef_[2]:.2f}")

# Show the dataset
#st.subheader('Sample Training Data')
#st.dataframe(df)

# Visualization
st.subheader('Data Visualization')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Hours Studied vs Marks
axes[0].scatter(df['Hours_Studied'], df['Marks'])
axes[0].set_xlabel('Hours Studied')
axes[0].set_ylabel('Marks')
axes[0].set_title('Hours Studied vs Marks')

# Sleep Hours vs Marks
axes[1].scatter(df['Sleep_Hours'], df['Marks'])
axes[1].set_xlabel('Sleep Hours')
axes[1].set_ylabel('Marks')
axes[1].set_title('Sleep Hours vs Marks')

# Previous Score vs Marks
axes[2].scatter(df['Previous_Score'], df['Marks'])
axes[2].set_xlabel('Previous Score')
axes[2].set_ylabel('Marks')
axes[2].set_title('Previous Score vs Marks')

plt.tight_layout()
st.pyplot(fig)

# Instructions
st.subheader('How to Use')
st.markdown("""
1. Adjust the input parameters in the sidebar
2. The app will automatically update the prediction
3. View the model details and visualizations below
""")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Scikit-learn")
