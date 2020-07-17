from model import *
import os
import numpy as np 
import cv2
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt


def load():
    m = create_model(2, False)
    m = model_load_weights("/home/jayda960825/ManTraNet_2020/pretrained_weights/ManTraNet_Ptrain4.h5", m)
    m.eval()
    return m

def read_rgb_image( image_file ) :
    rgb = cv2.imread( image_file, 1 )[...,::-1]
    return rgb
    
def decode_an_image_array( rgb, manTraNet ) :
    rgb = np.transpose(rgb, (2, 0, 1))
    x = np.expand_dims( rgb.astype('float32')/255.*2-1, axis=0 )
    x = torch.from_numpy(x).to(0, dtype=torch.float32)
    t0 = datetime.now()
    with torch.no_grad():
        y = manTraNet(x)
    t1 = datetime.now()
    return y, t1-t0

def decode_an_image_file( image_file, manTraNet ) :
    rgb = read_rgb_image( image_file )
    mask, ptime = decode_an_image_array( rgb, manTraNet )
    return rgb, mask, ptime.total_seconds()

def pca_3d(input_img = None,
    input_mask = None,
    plot_range = None):
    global dev
    m = load()
    forged_file = input_img
    rgb, mask, ptime = decode_an_image_file(forged_file, m ) 
    h = mask.shape[2]
    mask = mask.cpu().numpy()
    fm = m.feature_map('clstm')
    fm = np.squeeze(fm, 0) # squeeze batch
    fm = np.transpose(fm, (1,2,0)) # (H)
    features = fm.reshape([fm.shape[0]*fm.shape[1], fm.shape[2]]) 
    pca = PCA(n_components = 3).fit_transform(features)
    #tsne = TSNE(n_components = 3).fit_transform(pca)
    d = 0
    if input_mask:
        mask = cv2.imread(input_mask, cv2.COLOR_BGR2GRAY)
        mask = mask/255
        if mask.shape[0] != h:
            dev.append(-1)
            d = 1
        mx, my = np.where(mask == 1)
        mask = mask.reshape([mask.shape[0]*mask.shape[1]])

    else:
        mask = mask.reshape([mask.shape[2]*mask.shape[3]])
    # np.save('/home/jayda960825/Desktop/pca.npy', pca) 
    # np.save('/home/jayda960825/Desktop/mask.npy', mask)
    # img = read_rgb_image(input_img)
    # img[mx,my,0] = 255
    # img[mx,my,1] = 0
    # img[mx,my,2] = 0
    # plt.subplot(111)
    # plt.imshow(img)
    # plt.show()
    if d == 0:
        try:
            m = np.where(mask > 0.5)[0]
            p = np.where(mask <= 0.5)[0]
            mmean = pca[m,:].mean(0)
            pmean = pca[p,:].mean(0)
            dev.append(np.linalg.norm(mmean - pmean))
        except IndexError:
            dev.append(-1)
    # xs = pca[:,0]
    # ys = pca[:,1]
    # zs = pca[:,2]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(mmean[0], mmean[1], mmean[2], color = '#FFFF33', marker='^', s = 1000)
    # ax.scatter(pmean[0], pmean[1], pmean[2], color = '#00FF00', marker='o', s = 1000)
    # ax.scatter(xs[p], ys[p], zs[p], color='b', marker='^')
    # ax.scatter(xs[m], ys[m], zs[m], color='r', marker='o')

    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

if __name__ == '__main__':
    dev = []
    from numpy import genfromtxt
    import pandas as pd
    my_data = genfromtxt('/home/jayda960825/Desktop/auc.csv')
    my_data = my_data[:,1]
    my_data = np.expand_dims(my_data, axis=1)
    for i in range(1,101,1):
        pca_3d(input_img = '/home/jayda960825/Documents/NIST/image/'+str(i)+'t.tif',
            input_mask = '/home/jayda960825/Documents/NIST/mask/'+str(i)+'forged.tif')
    a = np.c_[ my_data, np.array(dev) ]  
    pd_data = pd.DataFrame(a)
    pd_data.to_csv('pd_data.csv')
