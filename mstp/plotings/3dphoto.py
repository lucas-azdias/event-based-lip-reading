import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Mudar diretório de trabalho para o local do script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Carregar o array de eventos
events = np.load(r'mstp\data\DVS-Lip\train\london\0.npy')

# Extrair x, y, t dos eventos com polaridade positiva (p=1)
x_vals_0 = []
y_vals_0 = []
t_vals_0 = []
x_vals_1 = []
y_vals_1 = []
t_vals_1 = []

for event in events:
    t, x, y, p = event['t'], event['x'], event['y'], event['p']
    if p:  # considerar apenas eventos com polaridade positiva
        x_vals_1.append(x)
        y_vals_1.append(y)
        t_vals_1.append(t / 1e5)  # Normalizar t
    else:  # considerar apenas eventos com polaridade positiva
        x_vals_0.append(x)
        y_vals_0.append(y)
        t_vals_0.append(t / 1e5)  # Normalizar t

# Converter para arrays numpy
x_vals_0 = np.array(x_vals_0)
y_vals_0 = np.array(y_vals_0)
t_vals_0 = np.array(t_vals_0)
x_vals_1 = np.array(x_vals_1)
y_vals_1 = np.array(y_vals_1)
t_vals_1 = np.array(t_vals_1)

# Plotagem 3D
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_vals_0, t_vals_0, y_vals_0, s=1, c="blue")
scatter = ax.scatter(x_vals_1, t_vals_1, y_vals_1, s=1, c="red")

ax.set_xlabel('$x$')
ax.set_ylabel('$Time / 10^{5}$')
ax.set_zlabel('$y$')

plt.subplots_adjust(left=0, right=0.90, top=1, bottom=0.05)

plt.show()