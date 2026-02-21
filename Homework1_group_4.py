
import numpy as np
import matplotlib.pyplot as plt

# ---- f0 değerini buraya yaz ----
f0 = 95   # örnek

f1 = f0
f2 = f0 / 2
f3 = 10 * f0

# Sampling frequency (Nyquist'e uygun)
fs = 50 * f0

# Her sinyal için en az 3 periyot
T1 = 1/f1
T2 = 1/f2
T3 = 1/f3

t1 = np.arange(0, 3*T1, 1/fs)
t2 = np.arange(0, 3*T2, 1/fs)
t3 = np.arange(0, 3*T3, 1/fs)

x1 = np.sin(2*np.pi*f1*t1)
x2 = np.sin(2*np.pi*f2*t2)
x3 = np.sin(2*np.pi*f3*t3)

# ---- Grafikler ----
plt.figure(figsize=(8,10))

plt.subplot(3,1,1)
plt.plot(t1, x1)
plt.title("f1 = f0")

plt.subplot(3,1,2)
plt.plot(t2, x2)
plt.title("f2 = f0/2")

plt.subplot(3,1,3)
plt.plot(t3, x3)
plt.title("f3 = 10f0")

plt.tight_layout()
plt.show()

# ---- Toplam sinyal ----
# Aynı zaman ekseni için en küçük pencereyi kullanıyoruz
t_sum = np.arange(0, 3*T1, 1/fs)
x_sum = (np.sin(2*np.pi*f1*t_sum) +
         np.sin(2*np.pi*f2*t_sum) +
         np.sin(2*np.pi*f3*t_sum))

plt.figure()
plt.plot(t_sum, x_sum)
plt.title("Sum of Three Signals")
plt.show()