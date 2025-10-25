# Static Sign Language Translator
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/kimmartelolives/static-sign-language-translator)

This project provides a real-time static sign language translator that uses a webcam to detect and interpret hand gestures. It leverages machine learning to classify signs and provides audio feedback for the translated word or letter.

## Features

- **Real-time Hand Tracking**: Utilizes Google's MediaPipe library for robust hand detection and landmark extraction.
- **Custom Data Collection**: Includes a script to easily capture and label new sign language gestures, allowing you to expand the model's vocabulary.
- **Machine Learning Model**: Employs a `RandomForestClassifier` to learn and predict signs based on the collected hand landmark data.
- **Text-to-Speech (TTS) Output**: Vocalizes the predicted sign using Google's Text-to-Speech engine, providing instant audio feedback.

## How It Works

The project operates in three main stages:

1.  **Data Collection (`collect_data.py`)**: This script activates the webcam and captures the 21 3D hand landmarks provided by MediaPipe. When you press a key, it saves the coordinates of these landmarks, normalized relative to the wrist, into `sign_data.csv` with a user-provided label (e.g., "A", "hello", "thanks").

2.  **Model Training (`train_model.py`)**: After collecting data, this script loads the `sign_data.csv` dataset. It then trains a `RandomForestClassifier` on this data to learn the patterns associated with each sign. The resulting trained model is saved as `sign_model.joblib`.

3.  **Translation (`translate_and_speak.py`)**: This is the main application. It loads the pre-trained `sign_model.joblib`, processes the live webcam feed, extracts hand landmarks in real-time, and feeds them into the model to get a prediction. A smoothing buffer is used to stabilize the predictions. The final predicted label is then spoken aloud.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kimmartelolives/static-sign-language-translator.git
cd static-sign-language-translator
```

### 2. Install Dependencies

This project requires Python and several libraries. You can install them using pip:

```bash
pip install opencv-python mediapipe scikit-learn pandas gTTS pygame
```

## Usage

The repository comes with a pre-trained model that can recognize basic signs.

### 1. Run the Translator

To start translating signs immediately, run the main application script:

```bash
python translate_and_speak.py
```

Position your hand in front of the webcam. The application will draw the landmarks on your hand, display the predicted sign, and speak the translation.

### 2. (Optional) Train on New Signs

You can train the model to recognize your own custom signs by following these steps.

#### Step A: Collect Data

Run the data collection script. You will be prompted to enter a label for the sign you want to teach.

```bash
python collect_data.py
```

-   Enter a label (e.g., `A`, `B`, `1`, `hello`) and press Enter.
-   Position your hand to form the sign.
-   Press the `c` key to capture a sample. A 3-second countdown will begin before the capture.
-   Capture multiple samples (20-30 are recommended) in slightly different positions for better accuracy.
-   To add a new sign, press `n`, enter the new label, and repeat the capture process.
-   Press `q` to quit the script when you are done.

Your new data will be appended to `sign_data.csv`.

#### Step B: Train the Model

After collecting your data, run the training script to create a new model based on your dataset.

```bash
python train_model.py
```

This script will train a new `RandomForestClassifier` and overwrite the existing `sign_model.joblib` with your custom-trained model. You can then run `translate_and_speak.py` to use your new model.

## File Descriptions

-   `collect_data.py`: Script to capture and label hand gesture data from a webcam.
-   `train_model.py`: Script to train the machine learning model on the collected data.
-   `translate_and_speak.py`: The main application that performs real-time sign detection and translation with audio output.
-   `sign_data.csv`: The dataset containing normalized hand landmark coordinates and their corresponding labels.
-   `sign_model.joblib`: The pre-trained machine learning model file.
-   `tts_audio/`: Directory created at runtime to store the generated audio files for text-to-speech.
