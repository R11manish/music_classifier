import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from flask import Flask, render_template, request
import pickle



app = Flask(__name__)

with open('music_genre_classifier.pkl', 'rb') as file:
    loaded_dict = pickle.load(file)

svm = loaded_dict['model']
scaler = loaded_dict['scaler']


# Use the loaded model for predictions


# Home page
@app.route('/')
def index():
  return render_template('index.html')

# Genre prediction
@app.route('/predict', methods=['POST'])
def predict():
  # Get the uploaded audio file
  audio = request.files['audio']
  audio_path = os.path.join('uploads', audio.filename)
  audio.save(audio_path)

    # Load and preprocess the audio file
  y, sr = librosa.load(audio_path, duration=30)
  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
  features = np.mean(mfcc, axis=1)
  features = scaler.transform([features])

    # Make a prediction
  genre_pred = svm.predict(features)[0]
  print(genre_pred)

    # Delete the uploaded file
  os.remove(audio_path)

  return render_template('result.html', genre=genre_pred)


if __name__ == '__main__':
  app.run(debug=True)
