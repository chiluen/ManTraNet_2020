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

def pca_3d(input_img = '/home/jayda960825/Documents/NIST/image/1t.tif',
    input_mask = '/home/jayda960825/Documents/NIST/mask/1forged.tif',
    plot_range = [50000,60000]):
    m = load()
    forged_file = input_img
    rgb, mask, ptime = decode_an_image_file(forged_file, m ) 

    featex = m.feature_map()
    featex = np.squeeze(featex, 0) # squeeze batch
    featex = np.transpose(featex, (1,2,0)) # (H)
    features = featex.reshape([featex.shape[0]*featex.shape[1], 256]) # 從第2個維度開始

    #tsne = TSNE(n_components=2).fit_transform(features)
    pca = PCA(n_components = 3).fit_transform(features)

    mask = img_gray = cv2.imread(input_mask, cv2.COLOR_BGR2GRAY)
    mask = mask.reshape([mask.shape[0]*mask.shape[1]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    t0 = datetime.now()
    for i in range(plot_range[0],plot_range[1],1):
        if(mask[i] == 0):
            ax.scatter(pca[i,0], pca[i,1], pca[i,2], color='b', marker='^')
        else:
            ax.scatter(pca[i,0], pca[i,1], pca[i,2], color='r', marker='o')
    t1 = datetime.now()
    print('ax.scatter time : ', t1-t0)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

if __name__ == '__main__':
    pca_3d()