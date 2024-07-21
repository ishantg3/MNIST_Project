import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the images
X = np.vstack((X_train, X_test)).reshape(-1, 28*28)
y = np.concatenate((y_train, y_test))

# Check the shape of the data
st.write(f"Shape of X: {X.shape}")
st.write(f"Shape of y: {y.shape}")

# Normalize the features
scaler = StandardScaler()

try:
    X_sc = scaler.fit_transform(X)
    st.write("Scaling successful.")
except Exception as e:
    st.write(f"Error during scaling: {e}")

# Train the models
try:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000).fit(X_sc, y),
        "Decision Tree": DecisionTreeClassifier().fit(X_sc, y),
        "Random Forest": RandomForestClassifier().fit(X_sc, y)
    }
    st.write("Models trained successfully.")
except Exception as e:
    st.write(f"Error during model training: {e}")

# Define Streamlit app
st.title("MNIST Digit Classifier")
st.write("This app predicts the digit from a handwritten image")

# Create input fields for the user
st.write("Please upload a handwritten image from your machine")
uploaded_file = st.file_uploader("Upload a 28x28 grayscale image of a handwritten digit", type=["jpeg", "png", "jpg"])

if uploaded_file is not None:
    try:
        # Open and pre-process the uploaded image
        image = Image.open(uploaded_file).convert("L")  # L means convert to grayscale
        image = image.resize((28, 28))
        image_arr = np.array(image).reshape(1, -1)
        image_arr_sc = scaler.transform(image_arr)

        # Select the model
        selected_model = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree", "Random Forest"])

        # Predict the image
        model = models[selected_model]
        y_pred = model.predict(image_arr_sc)

        # Display the result
        st.write(f"Predicted Digit for this image is: {y_pred[0]}")
    except Exception as e:
        st.write(f"Error during prediction: {e}")