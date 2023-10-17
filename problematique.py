"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt
import numpy as np
from helpers.ImageCollection import ImageCollection
import helpers.ClassificationData as CD

#######################################
def problematique_APP2():
    images = ImageCollection()
    # Génère une liste de N images, les visualise et affiche leur histo de couleur
    # TODO: voir L1.E4 et problématique
    if True:
        # TODO L1.E4.3 à L1.E4.5
        # Analyser quelques images pour développer des pistes pour le choix de la représentation
        N = 12
        im_list = images.get_samples(N)
        #print(im_list)
        #400,500 -> foret
        #350 -> plage nuit
        #100 -> plage jour
        im_list = np.array([1,2,3,4,5,6,7,8,9,10])+100
        #rgb,lab,hvs = images.get_color_info(500)

        images.images_display(im_list)
        images.view_histogrammes(im_list)
        if 0:
            im_list = im_list+500
            images.images_display(im_list)
            images.view_histogrammes(im_list)

        #print(CD.)
    # TODO L1.E4.6 à L1.E4.8
    #images.generateRepresentation()
    plt.show()


######################################
if __name__ == '__main__':
    problematique_APP2()
