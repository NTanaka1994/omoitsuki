from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("hip_constellation_line_star.csv", encoding="shift-jis")
hip = df.values
px = []
py = []
pz = []
r = 1
for i in range(len(hip)):
    size = 1 / hip[i][8] * 20
    if 10 < size:
        size = 10
    if hip[i][8] < 0:
        size = 10
    a = (hip[i][1] + (hip[i][2] + hip[i][3] / 60) / 60) * 15 * np.pi / 180
    f = 1
    if hip[i][4] == 0:
        f = -1
    else:
        f = 1
    c = f * (hip[i][5] + (hip[i][6] + hip[i][7] / 60) / 60) * np.pi / 180
    px.append(r*np.cos(a)*np.cos(c))
    py.append(r*np.sin(a)*np.cos(c))
    pz.append(r*np.sin(c))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(px, py, pz)
plt.show()
