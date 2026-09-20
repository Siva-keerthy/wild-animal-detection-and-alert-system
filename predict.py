import numpy as np
import librosa
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime
from tensorflow.keras.models import load_model
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import pygame


# Load the trained model
model = load_model(r"C:/Users/user/Desktop/hardware/animal_sound_model.h5")

# Load the label encoder (you must have saved it earlier)
le = joblib.load(r"C:/Users/user/Desktop/hardware/label_encoder.pkl")
def record_audio(filename="recorded.wav", duration=15, fs=44100):
    print(f" Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    recording = recording / np.max(np.abs(recording))  # Normalize the recording
    write(filename, fs, (recording * 32767).astype(np.int16))  # Save as 16-bit PCM WAV
    print(f" Recording saved as {filename}")
    return filename

# Upload the audio file
#audio_path = r"C:/Users/Dell/Desktop/animal_sounds/monkeys/monkey-scream-6407 - Copy.wav" 
audio_path = record_audio(filename="real_audio.wav", duration=15)

import pytz
# Set your local timezone (change to your region if needed)
local_tz = pytz.timezone("Asia/Kolkata")  # Example: IST
upload_time = datetime.now(pytz.utc).astimezone(local_tz)


# Function to extract MFCC features from audio
def extract_features(file_path, max_len=15):
    try:
        audio, sr = librosa.load(file_path, duration=max_len)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

        mfcc_mean = np.mean(mfcc.T, axis=0)
        chroma_mean = np.mean(chroma.T, axis=0)
        contrast_mean = np.mean(contrast.T, axis=0)

        combined = np.hstack([mfcc_mean, chroma_mean, contrast_mean])
        return combined  
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

from datetime import datetime
import pandas as pd

# Initialize log for pattern analysis
try:
    detection_log = pd.read_csv("detection_log.csv")
except:
    detection_log = pd.DataFrame(columns=["timestamp", "label", "confidence"])

# Predict function with confidence graph
def predict_audio(file_path, model, label_encoder, threshold=0.5,upload_time=None):
    print("Available classes:", list(label_encoder.classes_))

    feature = extract_features(file_path)
    if feature is None:
        print(" Could not extract features.")
        return

    feature = feature.reshape(1, -1)
    probs = model.predict(feature)[0]

    # Decode top prediction
    top_idx = np.argmax(probs)
    top_label = label_encoder.inverse_transform([top_idx])[0]
    top_conf = probs[top_idx]

    # Combine wild animal probabilities
    wild_labels = ['monkeys', 'squirrels']
    wild_conf = 0
    for label in wild_labels:
        if label in label_encoder.classes_:
            idx = np.where(label_encoder.classes_ == label)[0][0]
            wild_conf += probs[idx]

    print(f"\n🔊 Top Prediction: {top_label} ({top_conf * 100:.2f}%)")
    print(f" Wild Animal Confidence (Monkey + Squirrel): {wild_conf * 100:.2f}%")
    # Log the top prediction with timestamp
    timestamp = upload_time.strftime('%H:%M:%S') if upload_time else datetime.now().strftime('%H:%M:%S')
    detection_log.loc[len(detection_log)] = [timestamp, top_label, top_conf]
    detection_log.to_csv("detection_log.csv", index=False)

    if wild_conf >= threshold:
        print("⚠️ Wild animal presence detected. Action required!")
        try:
           play_alert_sound()
        except Exception as e:
           print(f" Could not play alert sound: {e}")
    else:
        print(" No wild animal detected. No action required.")

    # Plot class probabilities as a confidence bar chart
    class_labels = list(label_encoder.classes_)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(class_labels, probs * 100, color=['orange' if lbl in wild_labels else 'green' for lbl in class_labels])
    plt.title('Confidence per Class')
    plt.ylabel('Confidence (%)')
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Annotate bars with confidence values
    for bar, prob in zip(bars, probs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{prob * 100:.1f}%", ha='center', fontsize=9)

    plt.show()
    # Pattern Analysis - Peak time of day for each animal
    detection_log["timestamp"] = pd.to_datetime(detection_log["timestamp"])
    detection_log["date"] = detection_log["timestamp"].dt.date
    detection_log["hour_min"] = detection_log["timestamp"].dt.strftime('%H:%M')

    for animal in wild_labels:
        print(f"\n Analyzing pattern for: {animal}")
        animal_data = detection_log[detection_log["label"] == animal]

        if animal_data.empty:
            print(f" No data found for {animal}.")
            continue

        # Group by date and 30-min range to get daily peaks
        animal_data["time_range"] = animal_data["timestamp"].dt.floor("30min")
        daily_peak_times = (
            animal_data.groupby(["date", "time_range"])
            .size()
            .reset_index(name="count")
            .sort_values(["date", "count"], ascending=[True, False])
            .drop_duplicates(subset=["date"])
        )

        # Find the most frequent peak time across all days
        peak_counts = daily_peak_times["time_range"].value_counts()
        most_common_time = peak_counts.idxmax()
        peak_count = peak_counts.max()

        time_str = most_common_time.strftime("%I:%M %p")
        tomorrow_time_str = (most_common_time + pd.Timedelta(days=1)).strftime("%I:%M %p")

        print(f"Most frequent {animal} visit time: {time_str} ({peak_count} days)")
        print(f" {animal.capitalize()}s likely to visit around {tomorrow_time_str} tomorrow.")

        # Plot only the most common peak times
        plt.figure(figsize=(8, 4))
        peak_counts_sorted = peak_counts.sort_index()
        peak_counts_sorted.index = peak_counts_sorted.index.strftime("%I:%M %p")
        peak_counts.sort_index().plot(kind="bar", color="tomato")
        plt.title(f" Peak {animal.capitalize()} Visit Times (Daily Peaks Only)")
        plt.xlabel("Time")
        plt.ylabel("Number of Days")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()



def play_alert_sound():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(r"C:/Users/user/Desktop/hardware/alert_sound.wav")  # or .mp3
        pygame.mixer.music.play()
        
        # Wait for the sound to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f" Could not play alert sound: {e}")

# Run the prediction
predict_audio(audio_path, model, le, threshold=0.5,upload_time=upload_time)



