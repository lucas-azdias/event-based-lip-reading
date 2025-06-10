import numpy as np
import matplotlib.pyplot as plt
import os

# Mudar para o diretório do script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Carregar o array de eventos
events = np.load(r'mstp\data\DVS-Lip\train\london\0.npy')

# Definir dimensões da imagem
height, width = 128, 128

# Obter os valores de tempo
t_values = events['t']
t_min, t_max = t_values.min(), t_values.max()

# Dividir o tempo em 5 intervalos
time_bins = np.linspace(t_min, t_max, num=5)

# Criar a figura
fig, axes = plt.subplots(2, 2, figsize=(5, 5))
axes = axes.flatten()  # Transforma em array 1D para iteração

for i in range(len(axes)):
    t_start, t_end = time_bins[i], time_bins[i+1]
    
    # Selecionar eventos nesse intervalo de tempo
    mask = (t_values >= t_start) & (t_values < t_end)
    events_slice = events[mask]
    
    # Criar frame vazio
    frame = np.zeros((height, width), dtype=np.int8)
    
    # Acumular eventos com polaridade
    for event in events_slice:
        t, x, y, p = event['t'], event['x'], event['y'], event['p']
        if 0 <= y < height and 0 <= x < width:
            # frame[y, x] += 1 if p else 0
            frame[y, x] += 1 if p else -1
    
    # Plotar o frame
    axes[i].imshow(frame, cmap='gray', aspect='equal', vmin=0, vmax=1)
    axes[i].axis('off')
    axes[i].set_title(f'Frame {i+1}', y=-0.15, fontsize=10)

plt.gca().set_frame_on(False)  # Remove a moldura
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margem
plt.tight_layout()
plt.show()
