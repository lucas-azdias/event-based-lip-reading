import numpy as np
import matplotlib.pyplot as plt
import os

# Change the working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Carregar o array de eventos
events = np.load(r'mstp\data\DVS-Lip\train\london\0.npy')

# Descobrir as dimensões da imagem
height, width = 128, 128

# Criar um frame vazio
frame = np.zeros((height, width), dtype=np.uint8)

# Acumular eventos com polaridade positiva
for event in events:
    t, x, y, p = event['t'], event['x'], event['y'], event['p']
    if 0 <= y < height and 0 <= x < width:
        frame[y, x] += 1

# Mostrar imagem sem bordas ou eixos
plt.figure(figsize=(5, 5))
plt.imshow(frame, cmap='gray', aspect="equal", vmin=0, vmax=1)
plt.axis('off')             # Remove os eixos
plt.gca().set_frame_on(False)  # Remove a moldura
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margem
plt.margins(0, 0)
plt.show()
