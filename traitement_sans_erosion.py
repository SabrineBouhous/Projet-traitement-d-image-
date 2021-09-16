#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.image import imread
import math
def histogram(img):
    imgBGris=img.convert("L")
    img1 = np.array(imgBGris)
    # 2D array is convereted to an 1D
    fl = img1.flatten()
    # histogram and the bins of the image are computed
    hist,bins = np.histogram(img1,256,[0,255])
    # cumulative distribution function is computed
    cdf = hist.cumsum()
    # places where cdf=0 is masked or ignored and
    # rest is stored in cdf_m
    #print(cdf)
    cdf_m = np.ma.masked_equal(cdf,0)
    # histogram equalization is performed
    num_cdf_m = (cdf_m - cdf_m.min())*255
    den_cdf_m = (cdf_m.max()-cdf_m.min())
    cdf_m = num_cdf_m/den_cdf_m
    # the masked places in cdf_m are now 0
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    # cdf values are assigned in the flattened array
    im2 = cdf[fl]
    # im2 is 1D so we use reshape command to
    # make it into 2D
    im3 = np.reshape(im2,img1.shape)
    # converting im3 to an image
    im4= Image.fromarray(im3) # Transformation du tableau en image PIL
    # saving im4
    im_col=im4.convert('RGB')
    im_col.save('imgEqualis.jpg')
    return im_col

import skimage.io
from skimage.io import imread #pour lire les images et les convertir en gris
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image




#applying greyscale method
#correlation fuction from scratch
def corr(img,mask):
    im1=skimage.io.imread('imgEqualis.jpg')
    img=skimage.color.rgb2gray(im1)
    row,col=img.shape
    m,n=mask.shape
    new=np.zeros((row+m-1,col+n-1))
    n=n//2
    m=m//2
    filtered_img=np.zeros(img.shape)
    new[m:new.shape[0]-m,n:new.shape[1]-n]=img
    for i in range(m,new.shape[0]-m):
        for j in range(n,new.shape[1]-n):
            temp=new[i-m:i+m+1,j-m:j+m+1]
            result=temp*mask
            filtered_img[i-m,j-n]=result.sum()
    return filtered_img
#gaussian filtre from scratch
def gaussian(m,n,sigma):
    gaussian=np.zeros((m,n))
    m=m//2
    n=n//2
    for x in range(-m,m+1):
        for y in range(-n,n+1):
            x1=sigma*(2*np.pi)**2
            x2=np.exp(-(x**2+y**2)/(2*sigma**2))
            gaussian[x+m,y+n]=(1/x1)*x2
    return gaussian

######################################   # Amélioration du contraste :
 ######################################

def Contrast(img):
    im = img.convert('L')
    # im is converted to an ndarray
    im1 =  np.array(im)
    # finding the maximum and minimum pixel values
    b = im1.max()
    a = im1.min()
    # converting im1 to float
    c = im1.astype(float)
    # contrast stretching transformation
    im2 = 255*(c-a)/(b-a)
    # im2 is converted from an ndarray to an image
    im2= im1.astype('uint8')
    imgContrast =Image.fromarray(im2)# Transformation du tableau en image PIL
    im_colll=imgContrast.convert('RGB')
    im_colll.save('imgStrech.jpg')
    return im_colll







######################################   # Segmentation par clustering :
 ######################################
#fonction pour lire l'image
def read_image():

    # loading the png image
    img = imread('imgStrech.jpg')

    # scaling it so that the values are small
    img = img / 255

    return img
#fonction pour initializer lmeans
def initialize_means(img, clusters):

    # reshaping it or flattening it into a 2d matrix
    points = np.reshape(img, (img.shape[0] * img.shape[1],
                                             img.shape[2]))
    m, n = points.shape

    # clusters is the number of clusters
    # or the number of colors that we choose.

    # means is the array of assumed means or centroids.
    means = np.zeros((clusters, n))

    # random initialization of means.
    for i in range(clusters):
        rand1 = int(np.random.random(1)*10)
        rand2 = int(np.random.random(1)*8)
        means[i, 0] = points[rand1, 0]
        means[i, 1] = points[rand2, 1]

    return points, means



#fonction pour calculer les distances
def distance(x1, y1, x2, y2):

    dist = np.square(x1 - x2) + np.square(y1 - y2)
    dist = np.sqrt(dist)

    return dist
#l'algoithme k means from scratch
def k_means(points, means, clusters):

    iterations = 10 # the number of iterations
    m, n = points.shape

    # these are the index values that
    # correspond to the cluster to
    # which each pixel belongs to.
    index = np.zeros(m)

    # k-means algorithm.
    while(iterations > 0):

        for j in range(len(points)):

            # initialize minimum value to a large value
            minv = 1000
            temp = None

            for k in range(clusters):

                x1 = points[j, 0]
                y1 = points[j, 1]
                x2 = means[k, 0]
                y2 = means[k, 1]

                if(distance(x1, y1, x2, y2) < minv):
                    minv = distance(x1, y1, x2, y2)
                    temp = k
                    index[j] = k

        for k in range(clusters):

            sumx = 0
            sumy = 0
            count = 0

            for j in range(len(points)):

                if(index[j] == k):
                    sumx += points[j, 0]
                    sumy += points[j, 1]
                    count += 1

            if(count == 0):
                count = 1

            means[k, 0] = float(sumx / count)
            means[k, 1] = float(sumy / count)

        iterations -= 1

    return means, index
#fonction pour appliquer la segmentation
def segmented_image(means, index, img):

    # recovering the segmented image by
    # assigning each pixel to its corresponding centroid.
    centroid = np.array(means)
    recovered = centroid[index.astype(int), :]

    # getting back the 3d matrix (row, col, rgb(3))
    recovered = np.reshape(recovered, (img.shape[0], img.shape[1],
                                                     img.shape[2]))

    # plotting the segmented image.
    recoverede = (recovered * 255).astype(np.uint8)
    imgg=Image.fromarray(recoverede)
    imgg.save('seg.jpg')
    return imgg


#fonction pour faire appel aux autres fonction du segmentation

def Segmentation(img):
    img = read_image()
    clusters = 8
    points, means = initialize_means(img, clusters)
    means, index = k_means(points, means, clusters)
    segmented_image(means, index, img)








    ######################################   # dilation eterosion(nous allons applique la dilation) :
 ######################################
def dilation(img):

    img2= cv2.imread(img,0)#Acquire size of the image
    p,q= img2.shape#Show the image
    imgDilate= np.zeros((p,q), dtype=np.uint8)#Define the structuring element
    SED= np.array([[0,1,0], [1,1,1],[0,1,0]])
    constant1=1#Dilation operation without using inbuilt CV2 function
    for i in range(constant1, p-constant1):
        for j in range(constant1,q-constant1):
            temp= img2[i-constant1:i+constant1+1, j-constant1:j+constant1+1]
            product= temp*SED
            imgDilate[i,j]= np.max(product)
    cv2.imwrite("Dilated.png", imgDilate)
    return imgDilate



#ersion
#import numpy as np
#import cv2 #pour lire et afficher les images la bib n'a pas été utiliser pour les fonctions prédéfinées
#import numpy as np
#import matplotlib.pyplot as plt#Read the image for erosion
#img1= cv2.imread("B.jpg",0)#Acquire size of the image
#m,n= img1.shape #Show the image
#plt.imshow(img1, cmap="gray")# Define the structuring element
# k= 11,15,45 -Different sizes of the structuring element
#k=15
#SE= np.ones((k,k), dtype=np.uint8)
#constant= (k-1)//2#Define new image
#imgErode= np.zeros((m,n), dtype=np.uint8)#Erosion withoutusing inbuilt cv2 function for morphology
#for i in range(constant, m-constant):
    #for j in range(constant,n-constant):
        #temp= img1[i-constant:i+constant+1, j-constant:j+constant+1]
        #product= temp*SE
        #imgErode[i,j]= np.min(product)
        #plt.imshow(imgErode,cmap="gray")
#cv2.imwrite("Eroded3.png", imgErode)

######################################   # Main(faire appelauxfonctions egalistation histogramme, lissage , améliorationducontraste,segmentation,dilation) :
 ######################################
def traitement(img):
    # Egalisation d'histo
    egalisation_histo=histogram(img)
     # Lissage
    g=gaussian(5,5,2)
    lissage=corr(egalisation_histo,g)
    img = Image.fromarray(np.uint8(lissage * 255))
     #Contrast
    Contraste= Contrast(img)
    #segmentation
    Segmentationn=Segmentation(Contraste)
    #dilation
    dilationn=dilation('seg.jpg')
    return dilationn

