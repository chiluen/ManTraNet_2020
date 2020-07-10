import os
import numpy as np 
import cv2
import requests
import sys
import torch

from PIL import Image
from io import BytesIO
from matplotlib import pyplot
from datetime import datetime 


def url_plot(url, model, xrange = None, yrange = None):
    
    m = model
    url = url
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)
    if img.shape[-1] > 3 :
        img = img[...,:3]
    ori = np.array(img)
    if xrange is not None :
        img = img[:,xrange[0]:xrange[1]]
    if yrange is not None :
        img = img[yrange[0]:yrange[1]]
    x = np.expand_dims( img.astype('float32')/255.*2-1, axis=0 )
    x = np.transpose(x, (0,3,1,2))
    x = torch.from_numpy(x).cuda()
    
    t0 = datetime.now()
    with torch.no_grad():
        pred = m(x)
    t1 = datetime.now()
    ptime = (t1-t0).total_seconds()
    pred_final = pred[0,0,...]
    pred_final = pred_final.detach().cpu().numpy()
    pyplot.figure( figsize=(15,5) )
    pyplot.title('Original Image')
    pyplot.subplot(131)
    pyplot.imshow( img )
    pyplot.title('Forged Image (ManTra-Net Input)')
    pyplot.subplot(132)
    pyplot.imshow( pred_final, cmap='gray' )
    pyplot.title('Predicted Mask (ManTra-Net Output)')
    pyplot.subplot(133)
    pyplot.imshow( np.round(np.expand_dims(pred_final,axis=-1) * img).astype('uint8'), cmap='jet' )
    pyplot.title('Highlighted Forged Regions')
    pyplot.suptitle('Decoded {} of size {} for {:.2f} seconds'.format( url, img.shape, ptime ) )
    pyplot.show()