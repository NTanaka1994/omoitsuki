import numpy as np
import matplotlib.pyplot as plt

def en(theta, cx, cy):
    x = np.array([])
    y = np.array([])
    for i in range(len(cx)):
        x = np.append(x, np.cos(theta)+cx[i])
        y = np.append(y, np.sin(theta)+cy[i])
        line = np.arange(0, 360.1, 0.1)
        plt.plot(np.cos(np.pi*line/180)+cx[i],np.sin(np.pi*line/180)+cy[i],color="#000000")#,color="#c1ab05")
    return x, y


kaku = np.array([0, 60, 120, 180, 240, 300, 360])
thet = np.pi * kaku / 180
x, y = en(thet, [0], [0])
x2 = np.array(y)
y2 = np.array(x)
for i in range(4):
    x, y = en(thet, x, y)
plt.savefig("FlowerofLife")
plt.show()
