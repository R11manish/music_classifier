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

# Set the path to the GTZAN dataset
dataset_path = 'genres'

# Define the list of music genres
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Initialize empty lists to store the features and labels
features = []
labels = []

# Extract features from audio files
for genre in genres:
    genre_path = os.path.join(dataset_path, genre)
    for filename in os.listdir(genre_path):
        audio_path = os.path.join(genre_path, filename)
        print(audio_path)
        y, sr = librosa.load(audio_path, sr=None, duration=30)  # Load audio file (30 seconds)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extract MFCC features
        features.append(np.mean(mfcc, axis=1))  # Compute mean of MFCC coefficients
        labels.append(genre)

# Convert features and labels to numpy arrays
X = np.array(features)
y = np.array(labels)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM classifier
svm = SVC()
svm.fit(X_train, y_train)

model_scaler_dict = {
    'model': svm,
    'scaler': scaler,
}
with open('music_genre_classifier.pkl', 'wb') as file:
  pickle.dump(model_scaler_dict, file)