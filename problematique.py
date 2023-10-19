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
    images = ImageCollection(load_all=True)
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    # TODO: voir L1.E4 et problématique
    print("xd")
    if True:
            idx = np.arange(0,980,1)
            images.get_edge(idx)
            data = images.nb_edges
            data_to_view = np.zeros((980, 3))
            images.count_rgb_pixel()
            images.count_lab_pixel()

            # data_to_view[:,0] = images.sum_red_rgb
            # data_to_view[:, 1] = images.sum_green_rgb
            data_to_view[:, 2] = images.sum_blue_rgb

            data_to_view[:,0] = images.sum_red_rgb
            data_to_view[:, 1] = images.nb_red_pixels_lab
            data_to_view[:, 2] = images.nb_edges


            an.view3D(data_to_view, images.target, "3D")

            # plt.figure(1)
            # bins = 50
            # range = [0,6e6]
            # plt.hist(images.nb_edges,bins=bins,range=range)
            # plt.title("all")
            #
            # plt.figure(2)
            # plt.hist(images.nb_edges[0:360],bins=bins,range=range)
            # plt.title("coast")
            #
            # plt.figure(3)
            # plt.hist(images.nb_edges[360:688], bins=bins,range=range)
            # plt.title("forest")
            #
            # plt.figure(4)
            # plt.hist(images.nb_edges[688:], bins=bins,range=range)
            # plt.title("street")
            #
            plt.show()
            print("Done")

    if False:
        print("xd")
        # TODO L1.E4.3 à L1.E4.5
        # Analyser quelques images pour développer des pistes pour le choix de la représentation
        #N = 12
        #im_list = images.get_samples(N)
        #print(im_list)
        #400,500 -> foret
        #350 -> plage nuit
        #100 -> plage jour
    if False:
        im_list = np.array([1,2,3,4,5,6,7,8,9,10])+100

        images.images_display(im_list)
        images.view_histogrammes(im_list)
        plt.show()
        #hist = np.histogram(mean_lab[:,1],bins=128,density=True)
        bins = 128
        n = 2
        range = [-300,300]
    if False:
        im_list = np.arange(688, 980, 1)
        std_rgb, std_lab, std_hsv, target = images.get_rgb_lab_hsv_std(im_list)
        plt.figure(1)
        plt.hist(std_rgb[:,n],bins=bins,range = range)
        plt.title('street')
        #
        im_list = np.arange(0, 360, 1)
        std_rgb, std_lab, std_hsv, target = images.get_rgb_lab_hsv_std(im_list)
        plt.figure(2)
        plt.hist(std_rgb[:, n], bins=bins,range = range)
        plt.title('coast')

        im_list = np.arange(360, 688, 1)
        std_rgb, std_lab, std_hsv, target = images.get_rgb_lab_hsv_std(im_list)
        plt.figure(3)
        plt.hist(std_rgb[:, n], bins=bins,range = range)
        plt.title('forest')
    if False:
        im_list = np.arange(0,980,1)

        data_to_view = np.zeros(((len(im_list)), 3))

        #moy_rgb = np.mean(images.std_rgb,axis=2)
        #moy_lab = np.mean(images.std_lab, axis=2)
        #moy_hsv = np.mean(images.std_hsv,axis=2)

        std_mul_rgb = images.std_rgb[:,0,0] + images.std_rgb[:,0,1] + images.std_rgb[:,0,2]
        std_mul_lab = images.std_lab[:, 0, 0] + images.std_lab[:, 0, 1] + images.std_lab[:, 0, 2]
        std_mul_hsv = images.std_hsv[:, 0, 0] + images.std_hsv[:, 0, 1] + images.std_hsv[:, 0, 2]

        data_to_view[:,0] = std_mul_rgb
        data_to_view[:,1] = std_mul_lab
        data_to_view[:,2] = std_mul_hsv

        an.view3D(data_to_view,images.target,"3D")
        print("Done")

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
