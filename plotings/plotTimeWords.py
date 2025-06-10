import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# Caminho base
base_dir = r'mstp/data/DVS-Lip'

# Encontrar todos os arquivos .npy recursivamente
npy_files = npy_files = list(Path(base_dir).rglob("*.npy"))

# Lista para armazenar todos os tempos em segundos
all_t_seconds = np.ndarray(shape=(len(npy_files), 1))

# Processar cada arquivo
for i, file_path in enumerate(npy_files):
    events = np.load(file_path)
    t_values = events['t']  # Tempo em microssegundos
    t_sec = np.max(t_values) / 1e6  # Converter para segundos
    all_t_seconds[i] = t_sec

# Apply scale to simulate ×10⁻³
scale = 1e-3
frequencies, bins = np.histogram(all_t_seconds, bins=20)

# Plot
plt.figure(figsize=(4, 3))
plt.bar(bins[:-1], frequencies * scale, width=(bins[1]-bins[0]), color='royalblue', edgecolor='black', alpha=0.8)

# Custom formatter for y-axis to show ×10⁻³
formatter = FuncFormatter(lambda x, _: f"{x:.1f}")
plt.gca().yaxis.set_major_formatter(formatter)

# Add ×10⁻³ multiplier as y-axis label suffix
plt.xlabel("Duração da palavra (segundos)", fontsize=12)
plt.ylabel("Frequência ($\\times 10^{-3}$)", fontsize=12)

# Remove top and right spines
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()