"""
This scripts executes several forms of artificial generated noise on an exemplary sample.
Via trackers the gradation of the noise can be set and the corresponding PSNR is calculated.
"""


### Imports ###
# Standard library imports
import os
import time
import sys

# Third party imports
import cv2
import numpy as np
from random import randint, random
import progressbar
import math


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2 )
    if mse == 0:
        mse = 1e-10
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def noisy(noise_typ, image, noise_factor):
    """
    image : ndarray Input image data. Will be converted to float.
    noise_typ : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
    """
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = noise_factor
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p-perchannel":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = noise_factor
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "s&p-perpixel":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = noise_factor
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.shape[0] * image.shape[1] * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape[0:2]]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.shape[0] * image.shape[1]  * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape[0:2]]
        out[tuple(coords)] = 0
        return out
    elif noise_typ == "poisson":
        if noise_factor == 0:
            return image
        vals = len(np.unique(image)) / noise_factor
        vals = 2 ** np.ceil(np.log2(vals)) 
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy
    elif noise_typ =="blur":
        noise_factor = int(noise_factor)
        return cv2.GaussianBlur(image,(noise_factor,noise_factor),0)



def nothing(x):
    pass

# Create a black image, a window

if __name__ =='__main__':

    img = cv2.imread('./sample.jpg')
    img = cv2.resize(img,(640,360))

    blur = cv2.GaussianBlur(img,(1,1),0)


    merge = np.vstack((img, blur))

    cv2.imshow('blured_image', merge)

    # create trackbars for color change
    cv2.createTrackbar('Blurring','blured_image',0,100,nothing)
    cv2.createTrackbar('Salt & Pepper Noise', 'blured_image', 0, 100,nothing)
    cv2.createTrackbar('Gauss Noise', 'blured_image', 0, 100,nothing)
    cv2.createTrackbar('Poisson Noise', 'blured_image', 0, 100,nothing)

    widgets = [progressbar.Counter(format='%(value)02d von %(max_value)02d Herzen '), progressbar.AnimatedMarker(markers='|/-\\'[::-1]), progressbar.Bar(marker='❤')] # 
    progbar = progressbar.ProgressBar(widgets=widgets, max_value=30)
    progbar = progbar.start()

    while True:
        st = time.time()
        blur_kernel = cv2.getTrackbarPos('Blurring','blured_image') *2 + 1 #15
        sp_ratio = cv2.getTrackbarPos('Salt & Pepper Noise', 'blured_image') / 100 #6e-3
        gauss_ratio = cv2.getTrackbarPos('Gauss Noise', 'blured_image') /1000 #2e-3
        poiss_ratio = cv2.getTrackbarPos('Poisson Noise', 'blured_image')

        original = img.astype(np.float32) / 255 #np.copy(img)
        blur = noisy('blur', original, blur_kernel)
        salt_pepper = (noisy("s&p-perchannel", original, sp_ratio))
        gauss = (noisy("gauss", original, gauss_ratio))
        poisson = (noisy("poisson", original, poiss_ratio))
        speckle = (noisy("speckle", original, 0))

        font = 0
        fontsize = 1
        fontcolor = (0,0,0)
        thickness = 2
        lineType = cv2.LINE_AA
        text_pos = (10, 340)

        original_vis = original.copy()

        cv2.putText(original_vis, 'Original    PSNR: {0:.2f}'.format(psnr(original,original)), text_pos, font, fontsize, fontcolor, thickness, lineType)
        cv2.putText(blur, 'Blur    PSNR: {0:.2f}'.format(psnr(original,blur)), text_pos, font, fontsize, fontcolor, thickness, lineType)
        cv2.putText(salt_pepper, 'Salt & Pepper    PSNR: {0:.2f}'.format(psnr(original,salt_pepper)), text_pos, font, fontsize, fontcolor, thickness, lineType)
        cv2.putText(gauss, 'Gauss    PSNR: {0:.2f}'.format(psnr(original,gauss)), text_pos, font, fontsize, fontcolor, thickness, lineType)
        cv2.putText(poisson, 'Poisson    PSNR: {0:.2f}'.format(psnr(original,poisson)), text_pos,font, fontsize, fontcolor, thickness, lineType)
        cv2.putText(speckle, 'Speckle    PSNR: {0:.2f}'.format(psnr(original,speckle)), text_pos,font, fontsize, fontcolor, thickness, lineType)

        vert1 = np.vstack((original_vis, blur))
        vert2 = np.vstack((gauss, salt_pepper))
        vert3 = np.vstack((poisson, speckle))
        merge = np.hstack((vert1,vert2,vert3))



        cv2.imshow('blured_image', merge)
        
        cv2.waitKey(1)

        progbar.update(int(1/(time.time()- st)), force=True)