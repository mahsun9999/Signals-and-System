import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

# ======================
# LOAD AUDIO
# ======================
audio_path = "speech.wav"

signal, fs = librosa.load(audio_path, sr=None)
signal = signal / np.max(np.abs(signal))

# ======================
# PARAMETERS
# ======================
frame_length = int(0.02 * fs)
hop_length = int(0.01 * fs)

# ======================
# FEATURES
# ======================
rms = librosa.feature.rms(
    y=signal,
    frame_length=frame_length,
    hop_length=hop_length
)[0]

zcr = librosa.feature.zero_crossing_rate(
    signal,
    frame_length=frame_length,
    hop_length=hop_length
)[0]

# ======================
# ROBUST THRESHOLD (DÜZELTİLDİ)
# ======================
# Sesi ezbere yüzdelik dilimle kesmek yerine, maksimum sesin %5'ini eşik alıyoruz.
# Eğer çok sessiz kısımlar kalıyorsa 0.05'i 0.10 yapabilirsin. Çok kesiliyorsa 0.02 yapabilirsin.
max_energy = np.max(rms)
threshold = max_energy * 0.05 

speech_mask = rms > threshold

# ======================
# HANGOVER (DÜZELTİLDİ)
# ======================
hangover = 3

# Kendi kendini tetikleyip tüm dosyayı True yapmaması için kopya üzerinden gidiyoruz.
smoothed_mask = speech_mask.copy()

for i in range(hangover, len(speech_mask)):
    if not speech_mask[i] and np.any(speech_mask[i-hangover:i]):
        smoothed_mask[i] = True

speech_mask = smoothed_mask

# ======================
# EXTRACT SPEECH (DÜZELTİLDİ)
# ======================
speech_signal = []

for i in range(len(speech_mask)):
    if speech_mask[i]:
        start = i * hop_length
        # Sinyalin sonunu aşmaması için min() eklendi
        end = min(start + hop_length, len(signal)) 
        
        speech_signal.extend(signal[start:end])

speech_signal = np.array(speech_signal)

sf.write("speech_only.wav", speech_signal, fs)

# ======================
# VOICED / UNVOICED
# ======================
energy_threshold = np.median(rms)
zcr_threshold = np.median(zcr)

classification = []

for i in range(len(rms)):
    if not speech_mask[i]:
        classification.append(0)
    else:
        if rms[i] > energy_threshold and zcr[i] < zcr_threshold:
            classification.append(1)
        else:
            classification.append(2)

classification = np.array(classification)

# ======================
# PLOT
# ======================
time = np.arange(len(signal)) / fs

plt.figure(figsize=(12,10))

plt.subplot(3,1,1)
plt.plot(time, signal)
plt.title("Original Signal")

plt.subplot(3,1,2)
plt.plot(rms, label="RMS Energy", color='orange')
plt.plot(zcr, label="ZCR", color='green', alpha=0.7)
plt.axhline(threshold, color='red', linestyle='--', label="RMS Threshold (Kesme Noktası)")
plt.legend()

plt.subplot(3,1,3)
plt.plot(classification, color='purple')
plt.title("0=Silence 1=Voiced 2=Unvoiced")

plt.tight_layout()
plt.show()

# ======================
# COMPRESSION (DÜZELTİLDİ)
# ======================
original_duration = len(signal) / fs
new_duration = len(speech_signal) / fs

# Yüzdelik hesaplar netleştirildi
compression_saving = (1 - new_duration / original_duration) * 100
remaining_ratio = (new_duration / original_duration) * 100

print("\n----- RESULTS -----")
print(f"Original Duration: {original_duration:.2f} sec")
print(f"Speech Duration  : {new_duration:.2f} sec")
print(f"Space Saved (Ne Kadar Küçüldü): % {compression_saving:.2f}") 
print(f"Remaining   (Ne Kadarı Kaldı) : % {remaining_ratio:.2f}")