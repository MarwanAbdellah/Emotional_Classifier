# Emotional Classifier

Emotional Classifier is a deep learning-based facial emotion recognition system trained on the FER-2013 dataset.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage

Clone the repository and navigate to the project directory:

\`\`\`bash
git clone https://github.com/MarwanAbdellah/Emotional_Classifier.git
cd Emotional_Classifier
\`\`\`

### Train the Model (Optional)

If you want to train the model from scratch, run:

\`\`\`bash
python train.py
\`\`\`

### Run the Web Application

To start the Streamlit app:

\`\`\`bash
streamlit run app.py
\`\`\`

Then, upload an image to get an emotion prediction.

## Features

- **Deep Learning-based Emotion Recognition**: Uses a CNN model to classify facial expressions.
- **Pretrained Model Support**: The model is trained on 28,709 images and tested on 3,589 images, achieving a test accuracy of 64.78%.
- **Interactive Web App**: Users can upload images through a Streamlit-based interface and receive real-time emotion predictions.
- **Data Preprocessing**: Images are resized, normalized, and augmented to improve model performance.

## Technologies Used

- **Python**
- **TensorFlow/Keras**
- **OpenCV**
- **NumPy & Pandas**
- **Matplotlib & Seaborn**
- **Streamlit**

## Dataset

The **FER-2013 dataset** was used for training and evaluation. It consists of:

- **Training Set**: 28,709 images
- **Testing Set**: 3,589 images

## Model Performance

The trained model achieved:

- **Training Accuracy**: ~70%
- **Test Accuracy**: 64.78%
- **Loss Reduction**: Improved with data augmentation techniques.

## Future Improvements

- Implement real-time emotion detection via webcam.
- Enhance model performance with transfer learning.
- Increase dataset size to improve accuracy.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Project Link

[GitHub Repository](https://github.com/MarwanAbdellah/Emotional_Classifier)

---

🎯 **Author**: Marwan Abdellah
