import numpy as np
import matplotlib.pyplot as plt
import os

# Mudar para o diretório do script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Carregar o array de eventos
events = np.load(r'mstp\data\DVS-Lip\train\london\0.npy')

# Definir dimensões da imagem
height, width = 128, 128

split_like_dataset = False
if not split_like_dataset:
    # Obter os valores de tempo
    t_values = events['t']
    t_min, t_max = t_values.min(), t_values.max()

    # Dividir o tempo em 5 intervalos
    time_bins = np.linspace(t_min, t_max, num=5)

    # Criar a figura
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
else:
    # Dividir em grupos de 30 eventos
    group_size = 30
    num_groups = len(events) // group_size
    groups = np.array_split(events[:num_groups * group_size], num_groups)

    # Criar a figura
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
axes = axes.flatten()

for i in range(len(axes)):
    if not split_like_dataset:
        t_start, t_end = time_bins[i], time_bins[i+1]
        
        # Selecionar eventos nesse intervalo de tempo
        mask = (t_values >= t_start) & (t_values < t_end)
        events_slice = events[mask]
    else:
        events_slice = groups[i]

    # Obter coordenadas x, y e polaridade
    x = events_slice['x']
    y = events_slice['y']
    p = events_slice['p']
    
    # Cores: vermelho para p=1, azul para p=0
    colors = np.where(p == 1, 'red', 'blue')
    
    axes[i].scatter(x, y, c=colors, s=1)  # s define o tamanho dos pontos
    axes[i].invert_yaxis()  # Inverter eixo Y para coincidir com a imagem
    axes[i].set_xlim(0, width)
    axes[i].set_ylim(0, height)
    axes[i].axis('off')
    axes[i].invert_yaxis()
    axes[i].set_title(f'Frame {i+1}', y=-0.15, fontsize=10)

plt.tight_layout()
plt.show()
