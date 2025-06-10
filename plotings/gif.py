import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import os

# Change the working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configuration
folder_path = r'solution\DVS-Lip\train\london'
num_files = 150
height, width = 128, 128  # Adjust if needed

# List to store frames
frames = []

for i in range(num_files):
    file_path = os.path.join(folder_path, f"{i}.npy")
    events = np.load(file_path)

    frame = np.zeros((height, width), dtype=np.uint8)
    
    for event in events:
        x, y, p = event['x'], event['y'], event['p']
        if 0 <= y < height and 0 <= x < width:
            frame[y, x] += 1  # You can also weight by polarity if needed

    # Convert to PIL image with colormap (optional)
    img = Image.fromarray((cm.gray(frame / frame.max() if frame.max() > 0 else frame)[:, :, :3] * 255).astype(np.uint8))

    frames.append(img)

# Save as animated GIF
gif_path = "output.gif"
frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)