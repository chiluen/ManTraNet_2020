from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import random
import numpy as np 
import cv2
import pandas as pd
class ImageAnnotations3D():
    def __init__(self, xyz, imgs, ax3d,ax2d, resize_size = 50):
        self.xyz = xyz
        if imgs[0].shape[0]!= resize_size:
            self.imgs = resize(imgs, resize_size)
        else:
            self.imgs = imgs
        self.ax3d = ax3d
        self.ax2d = ax2d
        self.annot = []
        for s,im in zip(self.xyz, self.imgs):
            x,y = self.proj(s)
            self.annot.append(self.image(im,[x,y]))
        self.lim = self.ax3d.get_w_lims()
        self.rot = self.ax3d.get_proj()
        self.cid = self.ax3d.figure.canvas.mpl_connect("draw_event",self.update)

        self.funcmap = {"button_press_event" : self.ax3d._button_press,
                        "motion_notify_event" : self.ax3d._on_move,
                        "button_release_event" : self.ax3d._button_release}

        self.cfs = [self.ax3d.figure.canvas.mpl_connect(kind, self.cb) \
                        for kind in self.funcmap.keys()]

    def cb(self, event):
        event.inaxes = self.ax3d
        self.funcmap[event.name](event)

    def proj(self, X):
        """ From a 3D point in axes ax1, 
            calculate position in 2D in ax2 """
        x,y,z = X
        x2, y2, _ = proj3d.proj_transform(x,y,z, self.ax3d.get_proj())
        tr = self.ax3d.transData.transform((x2, y2))
        return self.ax2d.transData.inverted().transform(tr)

    def image(self,arr,xy):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr, zoom=2)
        im.image.axes = self.ax3d
        ab = offsetbox.AnnotationBbox(im, xy, xybox=(-30., 30.),
                            xycoords='data', boxcoords="offset points",
                            pad=0.3, arrowprops=dict(arrowstyle="->"))
        self.ax2d.add_artist(ab)
        return ab

    def update(self,event):
        if np.any(self.ax3d.get_w_lims() != self.lim) or \
                        np.any(self.ax3d.get_proj() != self.rot):
            self.lim = self.ax3d.get_w_lims()
            self.rot = self.ax3d.get_proj()
            for s,ab in zip(self.xyz, self.annot):
                ab.xy = self.proj(s)


def resize(imgs, resize_size):
    im = []
    for i in range(len(imgs)):
        im.append(cv2.resize(imgs[i], (resize_size, resize_size), interpolation=cv2.INTER_CUBIC))
    return im

if __name__ == '__main__':
    t1 = pd.read_csv('pd_data.csv', index_col=0)
    auc = t1['auc']
    dist = t1['distance']
    fig = plt.figure(figsize=(8,5))
    plt.scatter(auc, dist, marker='s', s=50)

    for i, (x, y) in enumerate(zip(auc, dist)):
        plt.annotate(
            '(%s)' %(i),
            xy=(x, y),
            xytext=(0, -10),
            textcoords='offset points',
            ha='center',
            va='top')
    plt.xlabel('auc')
    plt.ylabel('distance')
    plt.xlim([0.45,1])
    plt.ylim([0,1])
    plt.show()