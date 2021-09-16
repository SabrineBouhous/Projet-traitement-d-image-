import cv2
from cv2 import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.image import imread
import math

def sift():
    # lecture image
    img = cv2.imread('Dilated.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # appel fonction sift
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img,None)
    descriptors = []
    keypoints_2=[]
    keypoints_bdd=[]
    path = 'bdd_nv'

    files = os.listdir(path)
    for name in os.listdir(path):
        # lecture des images
        image1 =cv2.imread(os.path.join(path,name))
        #image2 = cv2.imread('./cible/001_1_1.bmp')
        # conversions des images
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        #image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # appel de la fonction sift
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints_2, descriptors_2 = sift.detectAndCompute(image1, None)
        #keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)
        descriptors.append(descriptors_2)
        keypoints_bdd.append(keypoints_2)
     #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    match=[]
    for i in range(len(descriptors)):
        matches = bf.match(descriptors[i],descriptors_1)
        matches = sorted(matches, key = lambda x:x.distance)
        match.append(matches)

    #calcule du score de matching

    number_keypoints = 0
    if len(keypoints_1) <= len(keypoints_bdd[0]):
        number_keypoints = len(keypoints_1)
    else:
        number_keypoints = len(keypoints_bdd[0])


    for i in range(len(match)):
        if(number_keypoints < len(match[i])):
            number_keypoints=len(match[i])

    #le calcule de pourcentge des matches (toutes les imgges)



    #calcule du score de matching

    number_keypoints = 0
    if len(keypoints_1) <= len(keypoints_bdd[0]):
        number_keypoints = len(keypoints_1)
    else:
        number_keypoints = len(keypoints_bdd[0])


    for i in range(len(match)):
        if(number_keypoints < len(match[i])):
            number_keypoints=len(match[i])
    tab_matches=[]
    #le calcule de pourcentge des matches (toutes les imgages)
    for i in range(len(match)):
        #print("Le poucentage de l'image: ",i,"=", (len(match[i]) /number_keypoints) * 100, "%")
        match_calc=len(match[i]) / number_keypoints * 100
        tab_matches.append(match_calc)

    #trouver le match max et son index
    match_max=np.amax(tab_matches)
    index_max=np.argmax(tab_matches, axis=0)

    #trouver l'image qui correspendant a l'index de match_max :
    i=0
    for name in files:
        if i == index_max :
            image_match=cv2.imread("bdd_nv/"+name)
            nom_image_match=name
            print('c bon on a trouver l"image qui correspendant au max_match')
            break
        else:
            i=i+1

    ima_trouvée = Image.fromarray(image_match)
    ima_trouvée.save('image_match.jpg')
    
    return tab_matches
    #ima_trouvée = cv2.cvtColor(Image.fromarray(image_match), cv2.COLOR_BGR2GRAY)
    #img3 = cv2.drawMatches(img, keypoints_1, ima,keypoints_bdd[match_max],matches[ima[:20]], ima, flags=2)
    #plt.imshow(img3)
    #plt.show()
    
    
