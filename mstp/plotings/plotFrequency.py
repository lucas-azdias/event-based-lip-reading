import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# Caminho base
base_dir = r'mstp/data/DVS-Lip'

# Encontrar todos os arquivos .npy recursivamente
npy_files = npy_files = list(Path(base_dir).rglob("*.npy"))

# Lista para armazenar todos os tempos em segundos
amount_events = np.ndarray(shape=(len(npy_files), 1))

# Processar cada arquivo
for i, file_path in enumerate(npy_files):
    events = np.load(file_path)
    amount_events[i] = len(events['t'])

# Apply scale to simulate ×10⁻³
x_scale = 1e-4
y_scale = 1e-3
frequencies, bins = np.histogram(amount_events, bins=20)

# Plot
plt.figure(figsize=(4, 3))
plt.bar(bins[:-1] * x_scale, frequencies * y_scale, width=(bins[1] - bins[0]) * x_scale,
        color='royalblue', edgecolor='black', alpha=0.8)

# Axis labels
plt.xlabel("Quantidade de eventos ($\\times 10^{-4}$)", fontsize=12)
plt.ylabel("Frequência ($\\times 10^{-3}$)", fontsize=12)

# Format y-axis
formatter_y = FuncFormatter(lambda y, _: f"{y:.1f}")
plt.gca().yaxis.set_major_formatter(formatter_y)

# Format x-axis
formatter_x = FuncFormatter(lambda x, _: f"{x:.1f}")
plt.gca().xaxis.set_major_formatter(formatter_x)

# Remove top and right spines
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()