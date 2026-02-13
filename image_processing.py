from sklearn.cluster import KMeans
import numpy as np

def Laplacian(arr, diff=4):
    cpy = np.zeros((arr.shape[0]+2, arr.shape[1]+2, arr.shape[2]))
    dst = np.zeros((arr.shape[0]+2, arr.shape[1]+2, arr.shape[2]))
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                cpy[i+1][j+1][k] = arr[i][j][k]
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                if diff == 4:
                    dst[i+1][j+1][k] = cpy[i+1][j+2][k] + cpy[i+1][j][k] + cpy[i+2][j+1][k] + cpy[i][j+1][k] - 4 * cpy[i+1][j+1][k]
                elif diff == 8:
                    dst[i+1][j+1][k] = cpy[i+1][j+2][k] + cpy[i+1][j][k] + cpy[i+2][j+1][k] + cpy[i][j+1][k] + cpy[i+2][j+2][k] + cpy[i+2][j][k] + cpy[i][j+2][k] + cpy[i][j][k] - 8 * cpy[i+1][j+1][k]
    dst2 = dst[1:len(dst), 1:len(dst[0])-1, :]
    dst2 = np.clip(dst2, 0, 255).astype(np.uint8)
    return dst2

def Sobel(arr, vector="width"):
    cpy = np.zeros((arr.shape[0]+2, arr.shape[1]+2, arr.shape[2]))
    dst = np.zeros((arr.shape[0]+2, arr.shape[1]+2, arr.shape[2]))
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                cpy[i+1][j+1][k] = arr[i][j][k]
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                if vector == "width":
                    dst[i+1][j+1][k] = - cpy[i][j+2][k] -2 * cpy[i][j+1][k] - cpy[i][j][k] + cpy[i+2][j+2][k] + 2 * cpy[i+2][j+1][k] + cpy[i+2][j][k]
                elif vector == "height":
                    dst[i+1][j+1][k] = - cpy[i][j+2][k] -2 * cpy[i+1][j+2][k] - cpy[i+2][j+2][k] + cpy[i][j][k] + 2 * cpy[i+1][j][k] + cpy[i+2][j][k]
    dst2 = dst[1:len(dst), 1:len(dst[0])-1, :]
    dst2 = np.clip(dst2, 0, 255).astype(np.uint8)
    return dst2

def Filter(arr, case="median"):
    cpy = np.zeros((arr.shape[0]+2, arr.shape[1]+2, arr.shape[2]))
    dst = np.zeros((arr.shape[0]+2, arr.shape[1]+2, arr.shape[2]))
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                cpy[i+1][j+1][k] = arr[i][j][k]
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                if case == "median":
                    dst[i+1][j+1][k] = np.median([
                        cpy[i+2][j+2][k], cpy[i+2][j+1][k], cpy[i+2][j][k],
                        cpy[i+1][j+2][k], cpy[i+1][j+1][k], cpy[i+1][j][k],
                        cpy[i][j+2][k], cpy[i][j+1][k], cpy[i][j][k],
                    ])
                elif case == "average":
                    dst[i+1][j+1][k] = np.mean([
                        cpy[i+2][j+2][k], cpy[i+2][j+1][k], cpy[i+2][j][k],
                        cpy[i+1][j+2][k], cpy[i+1][j+1][k], cpy[i+1][j][k],
                        cpy[i][j+2][k], cpy[i][j+1][k], cpy[i][j][k],
                    ])
    dst2 = dst[1:len(dst), 1:len(dst[0])-1, :]
    dst2 = np.clip(dst2, 0, 255).astype(np.uint8)
    return dst2

def Contrast(arr):
    hist = np.zeros((arr.shape[2], 256))
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                hist[k][arr[i][j][k]] = hist[k][arr[i][j][k]] + 1
    csum = []
    for i in range(len(hist)):
        csum.append(np.cumsum(hist[i]))
    dst = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]))
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                dst[i][j][k] = int(255 * csum[k][arr[i][j][k]] / max(csum[k]))
        dst = np.clip(dst, 0, 255).astype(np.uint8)
    return dst            

def Edge_sharp(arr, alpha=1.0, diff=8):
    cpy = np.zeros((arr.shape[0]+2, arr.shape[1]+2, arr.shape[2]))
    dst = np.zeros((arr.shape[0]+2, arr.shape[1]+2, arr.shape[2]))
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                cpy[i+1][j+1][k] = arr[i][j][k]
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            for k in range(len(arr[i][j])):
                if diff == 8:
                    dst[i+1][j+1][k] = - (cpy[i+1][j+2][k] + cpy[i+1][j][k] + cpy[i+2][j+1][k] + cpy[i][j+1][k] + cpy[i+2][j+2][k] + cpy[i+2][j][k] + cpy[i][j+2][k] + cpy[i][j][k]) / 9 + 8 * cpy[i+1][j+1][k] / 9
                elif diff == 4:
                    dst[i+1][j+1][k] = - (cpy[i+1][j+2][k] + cpy[i+1][j][k] + cpy[i+2][j+1][k] + cpy[i][j+1][k]) / 5 + 4 * cpy[i+1][j+1][k] / 5
                dst[i+1][j+1][k] = cpy[i+1][j+1][k] - alpha * dst[i+1][j+1][k] * 9.0
    dst2 = dst[1:len(dst), 1:len(dst[0])-1, :]
    dst2 = np.clip(dst2, 0, 255).astype(np.uint8)
    return dst2

def Poster(arr, num=10):
    data = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            data.append(arr[i][j])
    model = KMeans(n_clusters=num)
    model.fit(data)
    y_pred = model.predict(data)
    colors = model.cluster_centers_
    dst = y_pred.reshape(arr.shape[0], arr.shape[1])
    dst2 = []
    for i in range(len(dst)):
        tmp = []
        for j in range(len(dst[i])):
            tmp2 = []
            for k in range(len(arr[i][j])):
                tmp2.append(int(colors[dst[i][j]][k]))
            tmp.append(tmp2)
        dst2.append(tmp)
    return dst2
