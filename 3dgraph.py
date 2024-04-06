import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

coords = np.linspace(-12, 12, 1000)
x, y = np.meshgrid(coords,coords)
z = np.sin((y*y+x*x)**0.5)/(y**2+x**2) ** 0.5
fig, axs=plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True, subplot_kw={"projection":"3d"})
surf = axs[0].plot_surface(x, y, z, rstride=5, cstride=5, cmap="rainbow")
fig.colorbar(surf, ax=axs[0], shrink=0.6)
plt.show()
