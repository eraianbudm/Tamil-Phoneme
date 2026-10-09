# Tamil Phoneme Feedback Pronunciation System

## Project Overview

The Tamil Phoneme Feedback Pronunciation System is a deep learning-based project designed to classify Tamil phonemes that can be difficult to distinguish, such as ல, ள, and ழ. It uses a Convolutional Neural Network (CNN) to analyze speech recordings and predict the corresponding phoneme.

The project includes a Gradio web interface that allows users to record or upload audio and view the predicted phoneme along with its confidence score.

## Features

- Classifies 12 Tamil phoneme categories.
- Processes audio recordings at a 16 kHz sample rate.
- Uses a custom CNN model built with TensorFlow and Keras.
- Provides an interactive web interface using Gradio.
- Displays the predicted phoneme and confidence score.

## Technologies Used

- Python
- TensorFlow and Keras
- Librosa
- NumPy
- Gradio

## Project Workflow

1. Collect and annotate Tamil speech recordings.
2. Preprocess and normalize the audio.
3. Divide audio into frames for model training.
4. Train a CNN model to classify phonemes.
5. Integrate the trained model with the Gradio web interface.
6. Display the predicted phoneme and confidence score.

## Dataset

The project uses a custom dataset of Tamil speech recordings with corresponding phoneme annotations.

**Google Drive Dataset:** [Access the Dataset](PASTE_YOUR_DATASET_DRIVE_LINK_HERE)

## Limitations

The model's validation performance is limited, so its predictions may not always identify the spoken phoneme correctly. The displayed confidence score represents the model's prediction confidence and should not be interpreted as a verified pronunciation-correctness score.

## Future Enhancements

- Expand the dataset with more speakers and recordings.
- Improve model generalization and classification performance.
- Enhance feedback for commonly confused Tamil phonemes.
- Improve the web interface and audio-processing pipeline.

## Author

Developed as an academic project on Tamil phoneme classification and pronunciation feedback.
Eraianbu D M
GitHub: https://github.com/eraianbudm
LinkedIn: https://www.linkedin.com/in/eraianbu-dm
