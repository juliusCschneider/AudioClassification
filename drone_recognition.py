import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D, Dropout

from datasets import load_dataset, Audio
import pandas as pd
from itertools import groupby



TARGET_RATE = 16000


ds = load_dataset("geronimobasso/drone-audio-detection-samples", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_RATE))
ds = ds.shuffle(seed=8).select(range(50000))  # take 2000 samples for speed
#in the native dataset I got a ratio of about 9:1 of positives vs negatives. I found out that a more equal distribution yields better results
df = ds.to_pandas()
positives = df[df["label"] == 1].sample(n=2000, random_state=42)
negatives = df[df["label"] == 0].sample(n=2000, random_state=42)
sample = pd.concat([positives, negatives]).sample(frac=1, random_state=42) 
neg = sample.iloc[0]
pos = sample.iloc[1]
print(sample.head())
print(neg)
print(pos)


def load_wav_16k_mono(row): #converts audio to 16 hertz and gives us one channel
    file_bytes = row["audio"]["bytes"]
    wav, sample_rate = tf.audio.decode_wav(
        contents = file_bytes,
        desired_channels = 1
    )
    wav = tf.squeeze(wav, axis=-1)  # remove channel dim
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in = sample_rate, rate_out=16000)

    return wav

wave = load_wav_16k_mono(pos)
nwave = load_wav_16k_mono(neg)

#plt.plot(wave.numpy(), label="Drone")
#plt.plot(nwave.numpy(), label="No Drone")
#plt.legend()
#plt.show()

def get_wave_and_sr(example): #because load16k_mono was not general enough
    audio = example['audio']
    if isinstance(audio,dict) and "array" in audio: #HF dict with "array" and "samling_rate"
        arr = audio["audio"]
        arr = np.asarray(arr, dtype = np.float32)
        wav = tf.convert_to_tensor(arr, dtype=tf.float32)
        wav = tf.squeeze(wav, axis = -1)

        sr = audio.get("sampling_rate", TARGET_RATE)

    elif isinstance(audio,dict) and "bytes" in audio:
        b = audio["bytes"]
        contents = tf.convert_to_tensor(b, dtype = tf.string)
        wav, sr = tf.audio.decode_wav(contents=contents, desired_channels = 1)
        wav = tf.squeeze(wav, axis = -1)
    
    elif isinstance(audio, dict) and "path" in audio: #if audio is given as path string
        path = audio["path"]
        contents = tf.io.read_file(path)
        wav, sr = tf.audio.decode_wav(contents=contents, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)

    wav = tf.cast(wav, tf.float32)
    return wav,sr

# --- New clip length: 0.5 seconds ---
CLIP_LEN = 8000  # 0.5 seconds at 16 kHz

# Update preprocess_example
def preprocess_example(example, target_rate=TARGET_RATE, clip_len=CLIP_LEN, frame_length=320, frame_step=32):
    wav, sr = get_wave_and_sr(example)

    if sr != target_rate:
        wav = tfio.audio.resample(wav, rate_in=sr, rate_out=target_rate)

    # Crop or pad to 0.5 seconds
    wav = wav[:clip_len]
    pad_len = tf.maximum(clip_len - tf.shape(wav)[0], 0)
    wav = tf.concat([wav, tf.zeros([pad_len], dtype=tf.float32)], axis=0)

    stft = tf.signal.stft(wav, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(stft)
    spectrogram = tf.math.log1p(spectrogram)

    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    label = int(example["label"]) if not isinstance(example["label"], tf.Tensor) else int(example["label"].numpy())

    return spectrogram, label


clip_lengths = []

for _, row in positives.iterrows():
    wav, sr = get_wave_and_sr(row)
    # Resample to 16 kHz if needed
    if sr != TARGET_RATE:
        wav = tfio.audio.resample(wav, rate_in=sr, rate_out=TARGET_RATE)
    clip_lengths.append(wav.shape[0])  # number of samples

clip_lengths_sec = [length / TARGET_RATE for length in clip_lengths]

avg_length_sec = np.mean(clip_lengths_sec)
print(f"Average clip length: {avg_length_sec:.2f} seconds")

""" plt.hist(clip_lengths_sec, bins=30, color='skyblue')
plt.xlabel("Clip length (seconds)")
plt.ylabel("Number of clips")
plt.title("Distribution of Training Clip Lengths")
plt.show()  """



def build_dataset_arrays(df_iter, verbose = True):
    specs = []
    labels = []
    for i, ex in enumerate(df_iter):
        if isinstance(ex, tuple):
            ex = ex[1]
        spec, lab = preprocess_example(ex)
        specs.append(spec)
        labels.append(lab)
        if verbose and (i + 1) % 200 == 0:
            print(f"Processed {i+1} examples")
    
    X = tf.stack(specs)
    y = tf.convert_to_tensor(labels, dtype = tf.int32)
    return X,y

X, y = build_dataset_arrays(sample.iterrows(), verbose=True)
""" 
print("X shape:", X.shape)
print("y shape:", y.shape)
first_spec = X[1401]
plt.figure(figsize=(5,3))
plt.imshow(tf.transpose(first_spec[:,:,0]))  # transpose if you want time on x-axis, freq on y-axis
plt.xlabel("Time bins")
plt.ylabel("Frequency bins")
plt.title("First spectrogram")
plt.show()

second_spec = X[1901]
plt.figure(figsize=(10,8))
plt.imshow(tf.transpose(second_spec[:,:,0]))  # transpose if you want time on x-axis, freq on y-axis
plt.xlabel("Time bins")
plt.ylabel("Frequency bins")
plt.title("First spectrogram")
plt.show() 
print(y[1401])

print(y[1901]) """

# --- Train/Test split ---
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
#   # --- CNN Model ---
 
""" model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(242, 161, 1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(32, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    GlobalAveragePooling2D(),       # replaces Flatten
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer="adam", 
    loss="binary_crossentropy", 
    metrics=["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
)

# Train
hist = model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test))

# Save
model.save("drone_model_4.keras")   """

model = load_model('drone_model_4.keras')


test_loss, test_acc, test_recall, test_precision = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.3f}")
print(f"Test Recall: {test_recall:.3f}")
print(f"Test Precision: {test_precision:.3f}") 

print(model.summary())



# need new load function because longer files are in mp3
def load_mp3_16k_mono(filename):
    res = tfio.audio.AudioIOTensor(filename)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis = 1) / 2 #mp3 has two channels (stereo) and we want to take the avg out of the two
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in = sample_rate, rate_out = 16000)
    return wav

DRONE_DETECTION_AUDIO = os.path.join('drone_audio', 'drone_recording_05.mp3') #minimal length has to be 0.5 seconds


wave = load_mp3_16k_mono(DRONE_DETECTION_AUDIO)
audio_slices = tf.keras.utils.timeseries_dataset_from_array(wave, wave, sequence_length = 48000, sequence_stride = 48000, batch_size = 1)
samples, index = audio_slices.as_numpy_iterator().next()
print(samples.shape)

# --- Preprocess long audio for 0.5-second slices ---
def preprocess_mp3(sample, index):
    sample = sample[0]
    wav = sample[:CLIP_LEN]
    pad_len = tf.maximum(CLIP_LEN - tf.shape(wav)[0], 0)
    wav = tf.concat([wav, tf.zeros([pad_len], dtype=tf.float32)], axis=0)

    stft = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(stft)
    spectrogram = tf.math.log1p(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)

    return spectrogram

# Slicing for long audio
audio_slices = tf.keras.utils.timeseries_dataset_from_array(
    wave, wave, sequence_length=CLIP_LEN, sequence_stride=CLIP_LEN, batch_size=1
)
audio_slices = audio_slices.map(preprocess_mp3)
audio_slices = audio_slices.batch(64)


for batch in audio_slices.take(1):
    print(batch.shape)
""" 
for i, batch in enumerate(audio_slices.take(3)):
    # batch has shape (batch_size, time, freq, 1)
    batch = batch.numpy()  # convert to numpy
    for j in range(batch.shape[0]):
        spec = batch[j]  # single slice
        plt.figure(figsize=(10,4))
        plt.imshow(spec[:,:,0].T, origin='lower', aspect='auto', cmap='magma')
        plt.title(f"Spectrogram of Slice {i*batch.shape[0] + j + 1}")
        plt.xlabel("Time bins")
        plt.ylabel("Frequency bins")
        plt.colorbar(format='%+2.0f dB')
        plt.show()
 """



predictions_on_slices = model.predict(audio_slices)
print("Predictions shape:", predictions_on_slices.shape)
print(predictions_on_slices)
results_mp3 = []
for result in predictions_on_slices:
    if result < 0.1: results_mp3.append(0)
    else: results_mp3.append(1)
print("--- Results from long mp3 file: ---")
print(results_mp3)

results_mp3 = [key for key, group in groupby(results_mp3)]
print(results_mp3)
calls = tf.math.reduce_sum(results_mp3).numpy()
print(calls)

raw_probs = predictions_on_slices.squeeze()  # shape (num_slices,)


'''since it is pretty unlikely that a drone appears/disappears in 0.5 seconds I decided to smoothe the output of the CNN to get better results.'''
def smooth_predictions(probs, window_size=5):
    return np.convolve(probs, np.ones(window_size)/window_size, mode='same')

def apply_hysteresis(smoothed_probs, low_th=0.3, high_th=0.6):
    state = 0
    binary_preds = []
    for p in smoothed_probs:
        if state == 0 and p > high_th:
            state = 1
        elif state == 1 and p < low_th:
            state = 0
        binary_preds.append(state)
    return np.array(binary_preds)

#smoothing of probabilities
smoothed_probs = smooth_predictions(raw_probs, window_size=5)

# Apply hysteresis to stabilize ON/OFF detection
binary_preds = apply_hysteresis(smoothed_probs, low_th=0.3, high_th=0.6)

# Debugging outputs
print("Smoothed probabilities:", smoothed_probs[:20])
print("Binary stable predictions:", binary_preds[:20])

# Replace results_mp3 with smoothed binary predictions
results_mp3 = binary_preds.tolist()


# Each slice is 0.5 seconds long
slice_duration = 0.5  # seconds
time_axis = np.arange(len(raw_probs)) * slice_duration

plt.figure(figsize=(12,4))
plt.plot(time_axis, raw_probs, marker='o', label="Raw")
plt.plot(time_axis, smoothed_probs, marker='x', label="Smoothed")
plt.xlabel("Time (s)")
plt.ylabel("Probability of Drone")
plt.title("Drone Detection Probability Over Time (Smoothed)")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()


test_preds = model.predict(X_test[:20])  # first 20 examples to check how the model performs on training data
print("Raw predictions:", test_preds.squeeze())

test_labels = y_test[:20].numpy()
print("True labels:", test_labels)

# Convert probabilities to 0 and 1
test_preds_binary = (test_preds > 0.5).astype(int).squeeze()
print("Predicted labels:", test_preds_binary)



