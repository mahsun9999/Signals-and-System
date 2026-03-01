
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
import sounddevice as sd

# ======================================================
# PARAMETERS
# ======================================================

fs = 44100
duration = 0.04  # 40 ms
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# ======================================================
# CHARACTER - FREQUENCY MAPPING
# ======================================================

low_freqs = [700, 750, 800, 850, 900, 950]
high_freqs = [1200, 1250, 1300, 1350, 1400]

characters = list("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ ")

mapping = {}
reverse_mapping = {}

index = 0
for lf in low_freqs:
    for hf in high_freqs:
        if index < len(characters):
            mapping[characters[index]] = (lf, hf)
            reverse_mapping[(lf, hf)] = characters[index]
            index += 1

# ======================================================
# ENCODING
# ======================================================

def encode_text(text):
    full_signal = []

    for char in text:
        char = char.upper()
        if char in mapping:
            f1, f2 = mapping[char]
            tone = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)
            full_signal.extend(tone)

    return np.array(full_signal, dtype=np.float32)

# ======================================================
# DECODING (FFT)
# ======================================================

def decode_signal(signal):
    window_size = int(duration * fs)
    decoded_text = ""
    prev_char = None

    for i in range(0, len(signal), window_size):
        segment = signal[i:i+window_size]

        if len(segment) < window_size:
            continue

        segment = segment * np.hamming(window_size)

        fft = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(fft), 1/fs)
        magnitude = np.abs(fft)

        positive_freqs = freqs[:len(freqs)//2]
        positive_magnitude = magnitude[:len(magnitude)//2]

        peak_indices = positive_magnitude.argsort()[-2:]
        detected_freqs = sorted([abs(positive_freqs[i]) for i in peak_indices])

        detected_low = min(detected_freqs)
        detected_high = max(detected_freqs)

        closest_low = min(low_freqs, key=lambda x: abs(x - detected_low))
        closest_high = min(high_freqs, key=lambda x: abs(x - detected_high))

        char = reverse_mapping.get((closest_low, closest_high), "")

        if char != prev_char:
            decoded_text += char
            prev_char = char

    return decoded_text

# ======================================================
# MAIN
# ======================================================

text = "MERHABA DUNYA"

print("Original Text:", text)

# Encode
signal = encode_text(text)

# Save WAV
write("output.wav", fs, signal)

# Play sound
sd.play(signal, fs)
sd.wait()

# Decode
fs_read, data = read("output.wav")
decoded_text = decode_signal(data)

print("Decoded Text:", decoded_text)

# ======================================================
# GRAPHS
# ======================================================

# ---- 1) Time Domain of One Character ----
char_example = "A"
f1, f2 = mapping[char_example]
single_tone = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

plt.figure()
plt.plot(t, single_tone)
plt.title(f"Time Domain Signal - Character {char_example}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# ---- 2) FFT Spectrum with Peak Marking ----
windowed = single_tone * np.hamming(len(single_tone))
fft = np.fft.fft(windowed)
freqs = np.fft.fftfreq(len(fft), 1/fs)
magnitude = np.abs(fft)

positive_freqs = freqs[:len(freqs)//2]
positive_magnitude = magnitude[:len(magnitude)//2]

peak_indices = positive_magnitude.argsort()[-2:]

plt.figure()
plt.plot(positive_freqs, positive_magnitude)
plt.scatter(positive_freqs[peak_indices],
            positive_magnitude[peak_indices],
            color='red')

plt.title(f"Frequency Spectrum - Character {char_example}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 2000)
plt.grid()
plt.show()

# ---- 3) Full Encoded Signal ----
plt.figure()
plt.plot(signal)
plt.title("Full Encoded Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()