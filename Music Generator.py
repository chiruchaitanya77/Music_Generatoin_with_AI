import os
import numpy as np
from mido import Message, MidiFile, MidiTrack
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Directory containing MIDI files
input_files = ["Presto.mid", "Slend.mid", "und mit .mid", "Forever Free.mid"]  # Replace with your file names

# Initialize an empty list to store notes from all files
all_notes = []

# Process each MIDI file in the directory
for file in input_files:
    mid = MidiFile(file)
    notes = []

    # Extract notes from the MIDI file
    for msg in mid:
        if not msg.is_meta and msg.channel == 0 and msg.type == "note_on":
            data = msg.bytes()
            notes.append(data[1])

    all_notes.extend(notes)  # Add notes from this file to the global list

# Apply Min-Max scaling to normalize the notes
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.array(all_notes).reshape(-1, 1))
all_notes = list(scaler.transform(np.array(all_notes).reshape(-1, 1)))

# Prepare features and labels for training
X = []
y = []
n_prev = 30  # Number of notes in a batch
for i in range(len(all_notes) - n_prev):
    X.append(all_notes[i : i + n_prev])
    y.append(all_notes[i + n_prev])

# Convert lists to numpy arrays for model training
X = np.array(X)
y = np.array(y)

# Save a test subset for prediction
X_test = X[-300:]
X = X[:-300]
y = y[:-300]

# Define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(n_prev, 1), return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.6))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.6))
model.add(Dense(1))
model.add(Activation("linear"))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss="mse", optimizer=optimizer)

# Define file path and ModelCheckpoint callback
filepath = "./CheckPoints/checkpoint_model_{epoch:02d}.keras"
os.makedirs(os.path.dirname(filepath), exist_ok=True)
model_save_callback = ModelCheckpoint(
    filepath,
    monitor="val_loss",  # Changed to "val_loss" as "val_acc" may not be available with "mse" loss
    verbose=1,
    save_best_only=False,
    mode="auto",
    save_freq="epoch"  # Save every epoch
)

# Train the model
model.fit(X, y, batch_size=32, epochs=5, verbose=1, callbacks=[model_save_callback])

# Make predictions
prediction = model.predict(X_test)
prediction = np.squeeze(prediction)
prediction = np.squeeze(scaler.inverse_transform(prediction.reshape(-1, 1)))
prediction = [int(i) for i in prediction]

# Saving the result to a new MIDI file
mid = MidiFile()
track = MidiTrack()
for note in prediction:
    msg_on = Message('note_on', channel=0, note=note, velocity=67, time=0)
    msg_off = Message('note_off', channel=0, note=note, velocity=67, time=64)
    track.append(msg_on)
    track.append(msg_off)

mid.tracks.append(track)
mid.save("LSTM_music.mid")