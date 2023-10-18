"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt
import numpy as np
from helpers.ImageCollection import ImageCollection
import helpers.analysis as an
import cv2
import helpers.ClassificationData as CD

#######################################
def problematique_APP2():
    images = ImageCollection()
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    # TODO: voir L1.E4 et problématique
    if True:
        # TODO L1.E4.3 à L1.E4.5
        # Analyser quelques images pour développer des pistes pour le choix de la représentation
        #N = 12
        #im_list = images.get_samples(N)
        #print(im_list)
        #400,500 -> foret
        #350 -> plage nuit
        #100 -> plage jour
        #im_list = np.array([1,2,3,4,5,6,7,8,9,10])+100
        im_list = np.arange(688,980,1)
        #rgb,lab,hvs = images.get_color_info()
        mean_rgb,mean_lab,mean_hsv,target = images.get_rgb_lab_hsv_mean(im_list)
        #an.view3D(mean_lab, target, "3D")
        #data_to_view = np.zeros(((len(im_list)),3))
        #data_to_view[:,0] = mean_rgb[:,0]
        #data_to_view[:,1] = mean_lab[:, 1]
        #data_to_view[:,2] =  mean_hsv[:, 0]
        #images.images_display(im_list)
        #images.view_histogrammes(im_list)

        #hist = np.histogram(mean_lab[:,1],bins=128,density=True)
        bins = 56
        n = 1
        # range = [-10,10]
        # plt.figure(1)
        plt.figure(1)
        plt.hist(mean_lab[:,n],bins=bins)
        plt.title('street')
        #
        im_list = np.arange(0, 360, 1)
        mean_rgb, mean_lab, mean_hsv, target = images.get_rgb_lab_hsv_mean(im_list)
        plt.figure(2)
        plt.hist(mean_lab[:, n], bins=bins)
        plt.title('coast')

        im_list = np.arange(360, 688, 1)
        mean_rgb, mean_lab, mean_hsv, target = images.get_rgb_lab_hsv_mean(im_list)
        plt.figure(3)
        plt.hist(mean_lab[:, n], bins=bins)
        plt.title('forest')
        # #an.view3D(data_to_view,target,"3D")
        # print("Done")

        if 0:
            im_list = im_list+500
            images.images_display(im_list)
            images.view_histogrammes(im_list)

        #print(CD.)

    #images.generateRepresentation()
    plt.show()


######################################
if __name__ == '__main__':
    problematique_APP2()
