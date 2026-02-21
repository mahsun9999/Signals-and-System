
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------------------
# Sampling Parameters
# -------------------------------
fs = 8000          # Sampling frequency (Nyquist'e uygun)
T = 0.3            # Duration (seconds)

# -------------------------------
# DTMF Frequency Table
# -------------------------------
dtmf = {
    "1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "A": (697, 1633),
    "4": (770, 1209), "5": (770, 1336), "6": (770, 1477), "B": (770, 1633),
    "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "C": (852, 1633),
    "*": (941, 1209), "0": (941, 1336), "#": (941, 1477), "D": (941, 1633),
}

# -------------------------------
# GUI Setup
# -------------------------------
root = tk.Tk()
root.title("DTMF Signal Generator")
root.geometry("900x600")

# -------------------------------
# Matplotlib Figure (Time Domain)
# -------------------------------
fig, ax = plt.subplots(figsize=(7,4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10)

line, = ax.plot([], [])
ax.set_title("Time Domain Signal")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_ylim(-1.2, 1.2)

# -------------------------------
# Tone Generation Function
# -------------------------------
def play_tone(key):
    f_low, f_high = dtmf[key]

    t = np.arange(0, T, 1/fs)

    signal = (
        np.sin(2 * np.pi * f_low * t) +
        np.sin(2 * np.pi * f_high * t)
    ) * 0.5   # normalization (clipping önleme)

    # Sound Output
    sd.play(signal, fs)

    # Update Graph
    line.set_data(t, signal)
    ax.set_xlim(0, 0.05)  # 50 ms göster (daha net görünüm)
    canvas.draw()

# -------------------------------
# Keypad Frame
# -------------------------------
frame = ttk.Frame(root)
frame.pack()

keys = [
    ["1", "2", "3", "A"],
    ["4", "5", "6", "B"],
    ["7", "8", "9", "C"],
    ["*", "0", "#", "D"]
]

for r in range(4):
    for c in range(4):
        key = keys[r][c]
        button = ttk.Button(
            frame,
            text=key,
            command=lambda k=key: play_tone(k),
            width=8
        )
        button.grid(row=r, column=c, padx=5, pady=5)

root.mainloop()