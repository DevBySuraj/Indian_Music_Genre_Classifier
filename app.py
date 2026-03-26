import os
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
import tensorflow as tf

#libraries for building the model
from tensorflow.keras.layers import BatchNormalization, Conv2D,MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Converting all the audio files into the .wav format
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment # will convert the format

from tensorflow.image import resize

from spleeter.separator import Separator
import subprocess

from flask import Flask,request,app,jsonify,url_for,render_template
import io

import shutil

app = Flask(__name__)

classes = ['bollypop', 'carnatic', 'ghazal', 'semiclassical', 'sufi']
model = tf.keras.models.load_model("indian_music_classifier_v2.keras")

@app.route('/')
def home():
    return render_template('home.html')

#spleeter conversion
def spleeter_conversion(input_audio_path, test_output):

    # We call Spleeter as a system command. This avoids the "Finalized Graph" error.
    # It starts Spleeter, does the work, and closes it completely.
    cmd = f'spleeter separate -p spleeter:4stems -o "{test_output}" "{input_audio_path}"'
    subprocess.run(cmd, shell=True, check=True)
    
    # Locate the 'other' stem
    base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
    other_stem_path = os.path.join(test_output, base_name, "other.wav")
    
    if not os.path.exists(other_stem_path):
        raise FileNotFoundError(f"Spleeter failed to create {other_stem_path}")


other_stem_path = r"C:\Users\Suraj\Desktop\Projects\Indian_Music_Genre_Classifier\test_output\bp01\other.wav"
#chunking part
def load_and_preprocess_data(other_stem_path, sample_set_rate = 22050, target_shape = (150,150), threshold = 0.01 ):
        # --- STEP 2: Load and Chunk ---

    try:
        audio_data, sample_rate = librosa.load(other_stem_path, sr = sample_set_rate)
    except Exception as e:
        print(f"Skipping corrupted files: {other_stem_path}: {e}")
        
    #define the duration of each chunk and overlap
    chunk_duration = 4
    overlap_duration = 2
    data = []

    #convert duration to sample
    chunk_samples = int(chunk_duration*sample_rate)
    stride = int((chunk_duration-overlap_duration)*sample_rate) ## movement(4-2) * sr -> 2 second movement
    
    # sliding window logic
    # Use range with stride to ensure 'start + chunk_samples' never exceeds len(y)
    for start in range(0, len(audio_data) - chunk_samples + 1, stride):
        end = start + chunk_samples
        chunk = audio_data[start:end]

        # QUALITY CHECK: Check if chunk has actual sound (RMS energy)
        rms = np.sqrt(np.mean(chunk**2))
        if rms > threshold: 

            # melspectrogram part, this is the matrix we are getting the 
            mel_spectrogram = librosa.feature.melspectrogram(y = chunk, sr = sample_rate, n_mels = 150) #calculating spectrogram by this
            mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref= np.max)


            # 2. APPLY NORMALIZATION (Min-Max Scaling)
            # This ensures the test data matches the 0-1 range of your training data
            mel_min = mel_spectrogram.min()
            mel_max = mel_spectrogram.max()
            if mel_max - mel_min > 0:
                mel_norm = (mel_spectrogram - mel_min) / (mel_max - mel_min)
            else:
                mel_norm = mel_spectrogram # Handle silence/constant signal

            mel_spectrogram = resize(np.expand_dims(mel_norm, axis = -1), target_shape).numpy()

            # appending the data to the lists
            data.append(mel_spectrogram)
                
    return np.array(data)
      

#predict function
def model_prediction(X_test, classes):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred , axis = 1)
    unique_elements, counts = np.unique(predicted_categories, return_counts = True)
    max_element = np.max(counts)
    index = unique_elements[counts == max_element][0]
    return index


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('audio_file')

    if not file or file.filename == '':
        return render_template("home.html", predicted_genre="Error: No file selected")
    
    # 1. Save the uploaded file physically (Spleeter needs a path)
    # input file saved in the temp upload folder
    upload_folder = "./temp_uploads"
    os.makedirs(upload_folder, exist_ok=True)
    temp_input_path = os.path.join(upload_folder, file.filename)
    file.save(temp_input_path)


    # folder where stems will go after conversion by the spleeter
    test_output = "./temp_stems"
    if os.path.exists(test_output):
        shutil.rmtree(test_output) # Clean previous tests
    
    try:
        # 2. Run Spleeter (using the saved file path)
        spleeter_conversion(temp_input_path, test_output)

        # 3. Locate the 'other.wav' specifically for this file
        base_name = os.path.splitext(file.filename)[0]
        actual_stem_path = os.path.join(test_output, base_name, "other.wav")

        # 4. Preprocess and Predict
        X_test = load_and_preprocess_data(other_stem_path=actual_stem_path)
        
        if X_test.size == 0:
            return render_template("home.html", predicted_genre="Error: Audio too quiet or short")

        prediction_idx = model_prediction(X_test, classes)
        result = classes[prediction_idx]

        return render_template("home.html", predicted_genre=f"The predicted genre is {result}")

    except Exception as e:
        return render_template("home.html", predicted_genre=f"Error: {str(e)}")
    
    finally:
        # Clean up the uploaded MP3 to save space
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

if __name__ == "__main__":
    app.run(debug=True, use_reloader = False)

