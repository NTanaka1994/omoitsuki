import matplotlib.pyplot as plt
import numpy as np

def neduko(cnt, ctr):
    if cnt == 0:
        return 0
    else:
        cnt = cnt - 1
        tmp = []
        for i in range(6):
            for j in range(len(ctr)):
                tmp.append([ctr[j][0]+np.cos(60*i*np.pi/180), ctr[j][1]+np.sin(60*i*np.pi/180)])
            plt.plot([ctr[:, 0],ctr[:, 0]+np.cos(60*i*np.pi/180)/2], [ctr[:, 1],ctr[:, 1]+np.sin(60*i*np.pi/180)/2], color="#000000")
            plt.plot([ctr[:, 0],ctr[:, 0]+np.cos((60*i+30)*np.pi/180)/np.sqrt(3)], [ctr[:, 1],ctr[:, 1]+np.sin((60*i+30)*np.pi/180)/np.sqrt(3)], color="#000000")
        neduko(cnt, np.array(tmp))

ax = plt.axes()
ax.set_facecolor("#FDE8E9")
neduko(cnt=4, ctr=np.array([[0, 0]]))
plt.show()
