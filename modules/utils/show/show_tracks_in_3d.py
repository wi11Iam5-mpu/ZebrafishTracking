import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import pandas as pd

global colors


def animate_scatters(iteration, data, scatters, tracks_len=1000):
    l = len(scatters) // 3
    start = max((iteration - tracks_len, 0))
    timetext.set_text(f'Frame {iteration}')
    for i, key in enumerate(data[iteration]):
        x_3d, y_3d, z_3d, xz_2d, xy_2d = np.array([]), np.array([]), np.array([]), \
                                         np.array([[], []]).reshape(-1, 2), np.array([[], []]).reshape(-1, 2)
        for index in range(start, iteration + 1):
            if data[index][key].size != 0:
                x_3d = np.append(x_3d, data[index][key][0][0:1])
                y_3d = np.append(y_3d, data[index][key][0][1:2])
                z_3d = np.append(z_3d, data[index][key][0][2:])

                xz_2d = np.append(xz_2d, np.hstack([data[index][key][0][0:1],
                                                    data[index][key][0][2:]]).reshape(-1, 2), axis=0)
                xy_2d = np.append(xy_2d, data[index][key][0][0:2].reshape(-1, 2), axis=0)
            else:
                x_3d = np.append(x_3d, np.array([]))
                y_3d = np.append(y_3d, np.array([]))
                z_3d = np.append(z_3d, np.array([]))

                xz_2d = np.append(xz_2d, np.array([[], []]).reshape(-1, 2), axis=0)
                xy_2d = np.append(xy_2d, np.array([[], []]).reshape(-1, 2), axis=0)

        scatters[i]._offsets3d = (x_3d, y_3d, z_3d)
        scatters[i + l]._offsets = xz_2d
        scatters[i + 2 * l]._offsets = xy_2d

    return scatters


save = True

cmaps = [("Perceptually Uniform Sequential", [
            "viridis", "plasma", "inferno", "magma"]),
         ("Sequential", [
            "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
            "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu",
            "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn"]),
         ("Sequential (2)", [
            "binary", "gist_yarg", "gist_gray", "gray", "bone", "pink",
            "spring", "summer", "autumn", "winter", "cool", "Wistia",
            "hot", "afmhot", "gist_heat", "copper"]),
         ("Diverging", [
            "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu",
            "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic"]),
         ("Qualitative", [
            "Pastel1", "Pastel2", "Paired", "Accent",
            "Dark2", "Set1", "Set2", "Set3",
            "tab10", "tab20", "tab20b", "tab20c"]),
         ("Miscellaneous", [
            "flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern",
            "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg", "hsv",
            "gist_rainbow", "rainbow", "jet", "nipy_spectral", "gist_ncar"])]



# Attaching 3D axis to the figure
fig = plt.figure()
# ax = p3.Axes3D(fig)

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(2, 2, 4)
ax3 = fig.add_subplot(2, 2, 2)
timetext = ax1.text(20, 0, 40, '')
# Initialize scatters
# np.random.seed(666)


raw_data = pd.read_csv(
    r'D:\Projects\FishTracking\for_release\ZebrafishTracking\sequences\outputs\04\tracks_3d_interpolated.csv')
raw_data.columns = ['frame', 'id', '3d_x', '3d_y', '3d_z']

ids = raw_data['id'].unique()
frames = raw_data['frame'].unique()
ids = [int(_id) for _id in ids]
frames = [int(_fr) for _fr in frames]
# frames = frames[:500]

colors = {_id: np.random.choice(range(256), size=3).reshape(1, -1) / 255 for _id in ids}
scatters1 = [ax1.scatter([], [], [], c=colors[_id], s=0.2) for _id in ids]
scatters2 = [ax2.scatter([], [], c=colors[_id], s=0.2) for _id in ids]
scatters3 = [ax3.scatter([], [], c=colors[_id], s=0.2) for _id in ids]

scatters = scatters1 + scatters2 + scatters3

iterations = len(frames)

data = []
for frame in frames:
    _data = raw_data[raw_data.frame == frame]
    if _data.size != 0:
        temp_dict = {}
        for _id in ids:
            _d = _data[_data.id == _id]
            if _d.size == 0:
                _d = np.array([])
            else:
                _d = _d[['3d_x', '3d_y', '3d_z']].values
            temp_dict[_id] = _d
        data.append(temp_dict)
        # del temp_dict

# Setting the axes properties
ax1.set_xlim3d([0, 30])
ax2.set_xlim([0, 30])
ax3.set_xlim([0, 30])
ax1.set_xlabel('X')

ax1.set_ylim3d([0, 30])
ax2.set_ylim([0, 30])
ax3.set_ylim([0, 30])
ax1.set_ylabel('Y')

ax1.set_zlim3d([0, 30])
ax1.set_zlabel('Z')

ax1.set_title('3D Space')
ax2.set_title('Side view')
ax3.set_title('Top view')


# Provide starting angle for the view.
ax1.view_init(25, 45)

ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                              interval=10, blit=False, repeat=True)

if save:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
    ani.save('3d-scatted-animated.mp4', writer=writer)

plt.show()
