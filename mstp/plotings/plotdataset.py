def plot_voxel_grid_all(voxel_grid_all):
    import matplotlib.pyplot as plt
    # voxel_grid_all shape: [T, 1, H, W] => squeeze channel
    voxel_grid_all = voxel_grid_all.squeeze(1)  # Now shape [T, H, W]

    T, H, W = voxel_grid_all.shape

    # Initialize lists for plotting
    red_x, red_y = [], []
    blue_x, blue_y = [], []
    print(T)

    for t in range(30):#T):
        grid = voxel_grid_all[t]

        # Where voxel value is significantly non-zero (you can tune the threshold)
        y_coords, x_coords = np.where(np.abs(grid) > 0.1)
        values = grid[y_coords, x_coords]

        for x, y, val in zip(x_coords, y_coords, values):
            if val > 0:
                red_x.append(x)
                red_y.append(y)
            elif val < 0:
                blue_x.append(x)
                blue_y.append(y)
        # break

    plt.figure(figsize=(5, 5))
    plt.scatter(red_x, red_y, color='red', label='Polarity +1', s=1)
    plt.scatter(blue_x, blue_y, color='blue', label='Polarity -1', s=1)
    plt.axis('off')             # Remove os eixos
    plt.gca().set_frame_on(False)  # Remove a moldura
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margem
    plt.margins(0, 0)
    plt.gca().invert_yaxis()
    plt.show()