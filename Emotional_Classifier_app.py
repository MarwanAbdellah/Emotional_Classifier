import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the emotion classification model
model = load_model('Emotions_classifier.h5')

# Emotion labels
emotions = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def preprocess_image(image):
    """Preprocess the image for the model."""
    image = image.resize((224, 224))  # Resize to match model input
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

# Streamlit app layout
st.title('Emotion Classification App')
st.text('Upload an image or take a picture to classify its emotion.')


# Option to upload an image or take a picture
option = st.radio("Choose an option:", ("Upload Image", "Take a Picture"))

image = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif option == "Take a Picture":
    captured_image = st.camera_input("Take a picture")
    if captured_image is not None:
        image = Image.open(captured_image)

if image is not None:
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.text("Classifying emotion...")

    # Preprocess and predict
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    emotion_index = np.argmax(prediction)
    emotion_label = emotions[emotion_index]

    # Display the predicted emotion
    st.markdown(f"### Predicted Emotion: **{emotion_label.capitalize()}**")

    # Container for buttons and response
    response_message = ""
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Correct Prediction", key="correct", use_container_width=True):
            response_message = '<div class="response-container"><p style="color: green; font-size: 18px;">Yay! We\'re glad it worked out!</p></div>'
            st.balloons()
    with col2:
        if st.button("Incorrect Prediction", key="incorrect", use_container_width=True):
            response_message = (
                '<div class="response-container"><p style="color: red; font-size: 18px;">Oops! Sorry about that. Our model is trained on a limited dataset and may not always be accurate. '
                'We\'re working hard to improve it. Thanks for understanding!</p></div>'
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # Display the response message
    if response_message:
        st.markdown(response_message, unsafe_allow_html=True)
