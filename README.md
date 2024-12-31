# 🩺 **Skin Tumor Classification Web Application**

## ❤️ **Overview**

This web application is designed to predict the type of skin tumor from an uploaded image. The model uses deep learning techniques built with **TensorFlow** and a trained CNN (Convolutional Neural Network) to classify skin tumors into various categories.

## 🔧 **Requirements**

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Streamlit
- Pillow
- OpenCV
- NumPy

You can install the required dependencies with the following command:

********************************************************************
pip install tensorflow opencv-python streamlit pillow numpy
********************************************************************

**2. Install the Requirements**
Make sure you have all the necessary dependencies by running:

********************************************************************
pip install -r requirements.txt
********************************************************************

This will install packages like Streamlit, TensorFlow, and others needed to run the app.

**3. Run the Streamlit App**

To start the app, use this command:
********************************************************************
streamlit run app.py
********************************************************************

**4. Upload an Image**

Once the app is running, it will prompt you to upload a .jpg or .png image. The app supports common image formats for skin tumor classification.

**5. View the Prediction**

After uploading your image, the app will display the classification result, showing which tumor category the image belongs to.

# 🧠 **Model Summary**

The model is a Convolutional Neural Network (CNN) designed to classify images of skin tumors. It was built using TensorFlow/Keras and trained on a dataset of skin tumor images. Here's a breakdown of the model:

Model Type: CNN (Convolutional Neural Network)
Input Shape: 224x224x3 (RGB image)
Output: 10 tumor categories with prediction scores
Loss Function: Categorical Crossentropy
Optimizer: Adam Optimizer
The model processes input images, applies data augmentation (like rotation and flipping), and outputs the predicted tumor category.

#🏷️ **Supported Categories**
The model is capable of predicting the following skin tumor categories:

Eczema
Melanoma
Atopic Dermatitis
Basal Cell Carcinoma (BCC)
Melocytic Nevi (NV)
Benign Keratosis-like Lesions (BKL)
Psoriasis
Seborrheic Keratoses
Tinea Ringworm
Warts Molluscum

# 🌐 **Web App Features**
The app offers the following key features:

User-friendly Interface: The app has a simple, intuitive design for easy image uploads.
Fast Prediction: The model classifies the tumor type in real-time after you upload an image.
High Accuracy: Built with a pre-trained TensorFlow model, ensuring reliable predictions.

#📊 **Model Performance**

Here’s how the model performs:

Accuracy: The percentage of correct predictions made by the model.
Loss: The error measure that shows how close or far the predictions are from the actual classes.
The model is trained with a skin tumor dataset and evaluated based on accuracy and loss metrics.

#🧑‍💻 **Code Explanation**

The code is designed to be modular and easy to understand. The key steps are:

1. Model Loading
The pre-trained TensorFlow model is loaded, which is capable of processing and classifying skin tumor images.

2. Data Augmentation
To improve model robustness, random transformations like width, height adjustments, rotation, and flipping are applied to the images before feeding them into the model.

3. Prediction
Once the image is processed, the model predicts the tumor type and returns the most likely category.

#📥 **Requirements**
To run the project, you need to have the following:

Python 3.6 or higher
Streamlit
TensorFlow/Keras
NumPy
Pillow
Matplotlib
Install the required dependencies with:

********************************************************************
pip install -r requirements.txt
********************************************************************
