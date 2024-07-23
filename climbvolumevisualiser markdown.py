# %% [markdown]
# Firstly install needed libraries

# %%
# %pip install SpeechRecognition 
# %pip install pycaw 
# %pip install comtypes 
# %pip install sounddevice 
# %pip install numpy 
# %pip install opencv-python 
# %pip install pygame 
# %pip install tensorflow 
# %pip install keras 
# %pip install transformers 
# %pip install joblib 
# %pip install fer


# %% [markdown]
# Step 2:
#     Import the libraries as per needed

# %%
import time
import csv
import cv2
import numpy as np
import sounddevice as sd
from threading import Thread
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import speech_recognition as sr
from tensorflow.keras.models import load_model
import pygame
import joblib
from transformers import pipeline

# %% [markdown]
# Initialise AI feature toggle

# %%
AI_features_enabled = True

# %% [markdown]
# Making the model

# %%
#Import some extra modules from the tensorflow.keras.layers library that help with making emotion detection models
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

# Create the model
model = Sequential()

model.add(Input(shape=(48,48,1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Load the weights
emotion_model_path = r"C:\Users\kccha\OneDrive\Desktop\Programming\Climb volume visualizer\model.h5"
model.load_weights(emotion_model_path)

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]










# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D

# # Create the model
# model = Sequential()

# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(7, activation='softmax'))

# # Load the weights
# emotion_model_path = r"C:\Users\kccha\OneDrive\Desktop\Programming\Climb volume visualizer\model.h5"
# model.load_weights(emotion_model_path)

# EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


# %% [markdown]
# Addition/designing of volume control function(s)

# %%
# Volume control functions
def get_volume():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    return volume.GetMasterVolumeLevelScalar()

def set_volume(volume_level):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMasterVolumeLevelScalar(volume_level, None)

# %% [markdown]
# Then, for the voice control functionality, we would design/create voice control functions

# %%
# Voice control
def voice_control():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        with mic as source:
            print("Listening for command...")
            audio = recognizer.listen(source)
            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"Recognized command: {command}")
                if "increase volume" in command:
                    volume = get_volume() * 100
                    set_volume(min(volume + 10, 100) / 100)
                elif "decrease volume" in command:
                    volume = get_volume() * 100
                    set_volume(max(volume - 10, 0) / 100)
                elif "set volume to" in command:
                    try:
                        level = int(command.split("set volume to")[1].strip().replace("%", ""))
                        set_volume(min(max(level, 0), 100) / 100)
                    except ValueError:
                        pass
                elif "toggle ai features" in command:
                    toggle_AI_features()
            except sr.UnknownValueError:
                print("Could not understand the audio")
            except sr.RequestError:
                print("Could not request results from the service")

# %% [markdown]
# We would create a function for ambient noise detection so it does volume adjustments as well as per environment eg becomes less louder if environment becomes quiet

# %%
# Ambient noise detection
def measure_noise_level(duration=1):
    def callback(indata, frames, time, status):
        volume_norm = np.linalg.norm(indata) * 10
        global noise_level
        noise_level = volume_norm

    with sd.InputStream(callback=callback):
        sd.sleep(duration * 1000)

def adjust_volume_based_on_noise():
    while True:
        if AI_features_enabled:
            measure_noise_level()
            volume = get_volume()
            if noise_level > 50:
                set_volume(min(volume + 0.1, 1.0))
            elif noise_level < 20:
                set_volume(max(volume - 0.1, 0.0))
        time.sleep(1)

# %% [markdown]
# We create a function for emotion detection

# %%
# Emotion detection
def monitor_user_reactions():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        global user_reaction, emotion
        user_reaction = False
        emotion = None
        if len(faces) > 0:
            user_reaction = True
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype("float") / 255.0
                roi_gray = np.expand_dims(roi_gray, axis=-1)
                roi_gray = np.expand_dims(roi_gray, axis=0)
                preds = model.predict(roi_gray)[0]
                emotion = EMOTIONS[preds.argmax()]
        if AI_features_enabled:
            print(f"User reaction: {'Detected' if user_reaction else 'Not detected'}, Emotion: {emotion}")
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def adjust_volume_based_on_reaction():
    while True:
        if AI_features_enabled and user_reaction:
            volume = get_volume()
            if emotion == "happy":
                set_volume(min(volume + 0.1, 1.0))
            elif emotion in ["sad", "angry"]:
                set_volume(max(volume - 0.1, 0.0))
        time.sleep(1)

# %% [markdown]
# We create a function to help load user preferences

# %%
# Load user preferences
def load_user_preferences():
    preferences = {}
    try:
        with open('user_preferences.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                activity, volume = row
                preferences[activity] = float(volume)
    except FileNotFoundError:
        pass
    return preferences

def save_user_preferences(preferences):
    with open('user_preferences.csv', mode='w') as file:
        writer = csv.writer(file)
        for activity, volume in preferences.items():
            writer.writerow([activity, volume])

user_preferences = load_user_preferences()


# %% [markdown]
# Design and function creation for activity based volume references 

# %%
# Activity-based volume recommendations
activity_volume_recommendations = {
    "playing song": 0.5,
    "watching movie": 0.7,
    "reading book": 0.3
}

def get_recommended_volume(activity):
    if activity in user_preferences:
        return user_preferences[activity]
    elif activity in activity_volume_recommendations:
        return activity_volume_recommendations[activity]
    else:
        return 0.5  # Default volume


# %% [markdown]
# Designing the visualizer, I want it to be man/men moving on a ladder and intend the legs too climbing so users can have good visualization on volume control as different body parts move, such as one leg going up to indicate like a 0.1db volume increase

# %%
# Volume visualizer with video
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption('Volume Visualizer')

# Load video
video_path = r"C:\Users\kccha\OneDrive\Desktop\Programming\Climb volume visualizer\child_climbing_ladder.mp4"
cap = cv2.VideoCapture(video_path)

# Set initial positions
target_volume = get_volume()  # Initial target volume

def draw_visualizer():
    screen.fill((0, 0, 0))

    # Calculate current volume and adjust video position
    current_volume = get_volume()
    height = int(current_volume * 300)
    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
        ret, frame = cap.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)  # Rotate frame to match Pygame orientation
    frame = pygame.surfarray.make_surface(frame)
    
    frame_rect = frame.get_rect(center=(300, 400 - height))
    
    screen.blit(frame, frame_rect)
    pygame.display.update()

def run_visualizer():
    global target_volume
    while True:
        current_volume = get_volume()
        if current_volume != target_volume:
            # Move men towards the target volume
            move_amount = 0.01
            if current_volume > target_volume:
                target_volume = min(target_volume + move_amount, 1.0)
            elif current_volume < target_volume:
                target_volume = max(target_volume - move_amount, 0.0)
            time.sleep(0.05)  # Adjust speed of movement
        draw_visualizer()
        time.sleep(0.1)


# %% [markdown]
# Let's add a toggle for the AI features

# %%
# AI Feature Toggle
def toggle_AI_features():
    global AI_features_enabled
    AI_features_enabled = not AI_features_enabled
    print(f"AI features {'enabled' if AI_features_enabled else 'disabled'}")

# %% [markdown]
# Let's bring things all together

# %%
# Combine all components
global noise_level, user_reaction, emotion
noise_level = 0
user_reaction = False
emotion = None

# Threads for each component
voice_thread = Thread(target=voice_control)
noise_thread = Thread(target=adjust_volume_based_on_noise)
reaction_thread = Thread(target=monitor_user_reactions)
volume_adjust_thread = Thread(target=adjust_volume_based_on_reaction)
visualizer_thread = Thread(target=run_visualizer)

# Start all threads
voice_thread.start()
noise_thread.start()
reaction_thread.start()
volume_adjust_thread.start()
visualizer_thread.start()

# Join threads (Keep the script running)
voice_thread.join()
noise_thread.join()
reaction_thread.join()
volume_adjust_thread.join()
visualizer_thread.join()


