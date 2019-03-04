#from PIL import Image
import os
import cv2
import numpy as np
from skimage.io import imread, imsave, imshow
from matplotlib import pyplot as plt



def extractImage(small,x,y,rect_width,rect_height):
    template=np.zeros((rect_height,rect_width),np.uint8)
    for h in range(x,x+rect_width):
        for v in range(y,y+rect_height):
            template[v-y][h-x]=small[v][h]
    return template

# filter contours
def getContours(contours, hierarchy, tImage, originalIm):
    lst_im=[]
    for idx in range(0, len(hierarchy[0])):
        #print(contours[idx])
        rect = x, y, rect_width, rect_height = cv2.boundingRect(contours[idx])
        # ratio of non-zero pixels in the filled region
        if rect_width>=1 and rect_height>=1:
            rgb=cv2.rectangle(tImage,(x,y),(rect_width+x,rect_height+y),(0,0,255),1)
            tmp=extractImage(originalIm,x,y,rect_width,rect_height)
            lst_im.append(tmp)
    return reversed(lst_im)

def main():

    large = cv2.imread('blob.png')

    # downsample and use it for processing
    rgb = cv2.pyrDown(large)
    # apply grayscale
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # morphological gradient
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,4))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, morph_kernel)
    # binarize
    _, bw = cv2.threshold(src=grad, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 4))
    # connect horizontally oriented regions
    connected = cv2.morphologyEx(bw, cv2.MORPH_OPEN, morph_kernel)
    # find contours
    # im2, contours, hierarchy = cv2.findContours(connected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(connected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sPath='./output'

    # Extract the list of segmented blocks
    lst=getContours(contours, hierarchy, rgb, small)

    folder = os.path.join(os.getcwd(), 'segments')
    os.path.isdir(folder) or os.mkdir(folder)


    for idx,itm in enumerate(lst):

           cv2.imwrite(os.path.join(sPath,str(idx)+'.jpg'),itm)
           if itm.size < 30:
              continue
           imsave(os.path.join(folder,str(idx)+'.jpg'),itm)


    cv2.imshow('',rgb)
    cv2.waitKey(0)


if __name__=='__main__':
    main()
