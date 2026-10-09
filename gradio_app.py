import gradio as gr
import numpy as np
import librosa
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("cnn_withoutsil_tamil_model.h5")

# Label mapping 
label_to_id = {
    "l": 0, "lx": 1, "zh": 2, "la": 3, "lxa": 4,
    "zha": 5, "laa": 6, "lxaa": 7, "zhaa": 8,
    "li": 9, "lxi": 10, "zhi": 11
}
id_to_label = {v: k for k, v in label_to_id.items()}

# Constants
SR = 16000
FRAME_SIZE = 400
HOP_SIZE = 160
ENERGY_THRESHOLD = 0.01  # ignore silence frames


def preprocess_audio(audio):

    if audio is None:
        return None

    # if file path (upload), load; if tuple, unpack numpy array
    if isinstance(audio, tuple):
        y = audio[1].astype(np.float32)
    else:
        y, _ = librosa.load(audio, sr=SR)

    # normalize
    y = np.nan_to_num(y)
    y = y / (np.max(np.abs(y)) + 1e-8)

    # framing
    frames = librosa.util.frame(y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE).T

    # remove low-energy frames (silence)
    energies = np.mean(np.square(frames), axis=1)
    frames = frames[energies > ENERGY_THRESHOLD]

    if len(frames) == 0:
        return None

    # expand for CNN input
    frames = np.expand_dims(frames, -1)
    return frames


def predict(audio):
    frames = preprocess_audio(audio)
    if frames is None:
        return "No speech detected", 0.0

    preds = model.predict(frames, verbose=0)
    mean_pred = np.mean(preds, axis=0)
    pred_id = np.argmax(mean_pred)
    confidence = mean_pred[pred_id]

    return f"Predicted Tamil Letter: {id_to_label[pred_id]}", f"Confidence: {confidence*100:.2f}%"


# Build Gradio UI
app = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="🎤 Record or Upload Tamil Letter"),
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Textbox(label="Confidence Level")
    ],
    title="🗣️ Tamil Phoneme Feedback Pronunciation System",
    description="Record or upload a Tamil letter pronunciation. The model predicts which Tamil letter sound you spoke."
)

if __name__ == "__main__":
    app.launch()
