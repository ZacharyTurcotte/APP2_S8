"""
Classe "ImageCollection" pour charger et visualiser les images de la problématique
Membres :
    image_folder: le sous-répertoire d'où les images sont chargées
    image_list: une énumération de tous les fichiers .jpg dans le répertoire ci-dessus
    images: une matrice de toutes les images, (optionnelle, changer le flag load_all du constructeur à True)
    all_images_loaded: un flag qui indique si la matrice ci-dessus contient les images ou non
Méthodes pour la problématique :
    generateRGBHistograms : calcul l'histogramme RGB de chaque image, à compléter
    generateRepresentation : vide, à compléter pour la problématique
Méthodes génériques : TODO JB move to helpers
    generateHistogram : histogramme une image à 3 canaux de couleurs arbitraires
    images_display: affiche quelques images identifiées en argument
    view_histogrammes: affiche les histogrammes de couleur de qq images identifiées en argument
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
import cv2
import time
from enum import IntEnum, auto

from skimage import color as skic
from skimage import io as skiio

import helpers.analysis as an


class ImageCollection:
    """
    Classe globale pour regrouper les infos utiles et les méthodes de la collection d'images
    """

    class imageLabels(IntEnum):
        coast = auto()
        forest = auto()
        street = auto()

    def __init__(self, load_img=None):
        # liste de toutes les images
        self.image_folder = r"data" + os.sep + "baseDeDonneesImages"
        self._path = glob.glob(self.image_folder + os.sep + r"*.jpg")
        image_list = os.listdir(self.image_folder)
        # Filtrer pour juste garder les images
        self.image_list = [i for i in image_list if '.jpg' in i] # filtre .jpeg
        self.image_list2 = []

        if load_img is None:
            self.image_list2 = self.image_list
        else:
            for i in range(len(load_img)):
                self.image_list2.append(self.image_list[i])



        self.target = []
        self.all_images_loaded = False
        self.images = []
        self.nb_edges = []
        self.nb_images = 0
        # Crée un array qui contient toutes les images
        # Dimensions [980, 256, 256, 3]
        #            [Nombre image, hauteur, largeur, RGB]
        if load_img is None:
            self.images = np.array([np.array(skiio.imread(image)) for image in self._path])
            self.all_images_loaded = True
        else:
            if type(load_img) == int:
                load_img = [load_img]
            for i in range(len(load_img)):
                self.images.append(skiio.imread(self.image_folder + os.sep + self.image_list[load_img[i]]))
        self.nb_images = len(self.images)
        self.labels = []

        for i in self.image_list2:
            if 'coast' in i:
                self.labels.append(ImageCollection.imageLabels.coast)
                self.target.append(0)
            elif 'forest' in i:
                self.labels.append(ImageCollection.imageLabels.forest)
                self.target.append(1)
            elif 'street' in i:
                self.labels.append(ImageCollection.imageLabels.street)
                self.target.append(2)
            else:
                raise ValueError(i)

        # calculate rgb, lab, hsv
        self.rgb = []
        self.lab = []
        self.hsv = []

        self.nb_blue_pixels_lab = np.zeros(self.nb_images)
        self.nb_red_pixels_lab = np.zeros(self.nb_images)
        self.nb_green_pixels_lab = np.zeros(self.nb_images)
        self.nb_yellow_pixels_lab = np.zeros(self.nb_images)

        self.sum_red_rgb = np.zeros(self.nb_images)
        self.sum_green_rgb = np.zeros(self.nb_images)
        self.sum_blue_rgb = np.zeros(self.nb_images)

        # calculate means
        self.mean_rgb = []
        self.mean_lab = []
        self.mean_hsv = []

        # calculate stds
        self.std_rgb = []
        self.std_lab = []
        self.std_hsv = []

        for i in range(len(self.images)):
            self.get_color_info(self.images[i])
            self.get_rgb_lab_hsv_mean(self.images[i])
            self.get_rgb_lab_hsv_std()

        self.mean_rgb = np.array(self.mean_rgb)
        self.mean_lab = np.array(self.mean_lab)
        self.mean_hsv = np.array(self.mean_hsv)

        self.std_rgb = np.array(self.std_rgb)
        self.std_lab = np.array(self.std_lab)
        self.std_hsv = np.array(self.std_hsv)

        self.mean_rgb = np.reshape(self.mean_rgb, (self.nb_images, 3))
        self.mean_lab = np.reshape(self.mean_lab, (self.nb_images, 3))
        self.mean_hsv = np.reshape(self.mean_hsv, (self.nb_images, 3))

        self.std_rgb = np.reshape(self.std_rgb, (self.nb_images, 3))
        self.std_lab = np.reshape(self.std_lab, (self.nb_images, 3))
        self.std_hsv = np.reshape(self.std_hsv, (self.nb_images, 3))

        self.red_green_diff = np.zeros(self.nb_images)
        self.red_blue_diff = np.zeros(self.nb_images)
        self.blue_green_diff = np.zeros(self.nb_images)


    def get_samples(self, N):
        return np.sort(random.sample(range(np.size(self.image_list, 0)), N))

    def count_rgb_pixel(self):
        for i in range(self.nb_images):
            rgb = np.reshape(self.rgb[i], (256 * 256, 3))
            self.sum_blue_rgb[i] = np.sum(rgb[:, 2])
            self.sum_red_rgb[i] = np.sum(rgb[:, 0])
            self.sum_green_rgb[i] = np.sum(rgb[:, 1])

    def count_lab_pixel(self):
        for i in range(self.nb_images):
            lab = np.reshape(self.lab[i], (256 * 256, 3))
            self.nb_green_pixels_lab[i] = np.sum(np.array(lab[:, 1]) <= 0, axis=0)
            self.nb_red_pixels_lab[i] = np.sum(np.array(lab[:, 1]) > 0, axis=0)
            self.nb_blue_pixels_lab[i] = np.sum(np.array(lab[:, 2]) <= 0, axis=0)
            self.nb_yellow_pixels_lab[i] = np.sum(np.array(lab[:, 2]) > 0, axis=0)

    def get_edge(self, view=False):

        for i in range(self.nb_images):
            grayscale = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)
            # cv2.imshow('Grayscale', grayscale)
            edges = cv2.Canny(grayscale, 100, 200)
            if view:
                plt.figure(2 * i)
                plt.imshow(edges, cmap='gray')
                plt.figure(2 * i + 1)
                plt.imshow(self.images[idx[i]])
            self.nb_edges.append(np.sum(edges))
        # plt.show()

    def get_cov(self):
        images = []
        for image in self.images:
            images.append(cv2.cvtColor(self.images[image], cv2.COLOR_BGR2GRAY))
        self.images = images

    def get_select(self, index):
        return index

    def generateHistogram(self, image, n_bins=256):
        # Construction des histogrammes
        # 1 histogram per color channel
        n_channels = 3
        pixel_values = np.zeros((n_channels, n_bins))
        for i in range(n_bins):
            for j in range(n_channels):
                pixel_values[j, i] = np.count_nonzero(image[:, :, j] == i)
        return pixel_values

    def generateRGBHistograms(self):
        """
        Calcule les histogrammes RGB de toutes les images
        """
        # TODO L1.E4.6 S'inspirer de view_histogrammes et déménager le code pertinent ici
        raise NotImplementedError()

    def generateRepresentation(self):
        # produce a ClassificationData object usable by the classifiers
        # TODO L1.E4.8: commencer l'analyse de la représentation choisie
        raise NotImplementedError()

    def images_display(self):
        fig2 = plt.figure()
        ax2 = fig2.subplots(self.nb_images, 1)
        for i in range(self.nb_images):
            if self.all_images_loaded:
                pass # on devrait pas print toutes les images
            else:
                im = self.images[i]
                ax2[i].imshow(im)

    def get_color_info(self, image):

        imageRGB = image
        imageLab = skic.rgb2lab(image)
        imageHVS = skic.rgb2hsv(image)

        self.rgb.append(imageRGB)
        self.lab.append(imageLab)
        self.hsv.append(imageHVS)

    def classify(self, idx):
        if 0 <= idx < 360:
            return 0
        elif 360 <= idx < 688:
            return 1
        else:
            return 2

    def get_rgb_lab_hsv_mean(self, image):

        mean_rgb = np.zeros((1, 3))
        mean_lab = np.zeros((1, 3))
        mean_hsv = np.zeros((1, 3))

        imageRGB = np.array(self.rgb[-1])
        imageLab = np.array(self.lab[-1])
        imageHVS = np.array(self.hsv[-1])

        imageRGB = imageRGB.reshape(256 * 256, 3)
        imageLab = imageLab.reshape(256 * 256, 3)
        imageHVS = imageHVS.reshape(256 * 256, 3)

        mean_rgb[0, 0] = np.mean(imageRGB[:, 0])
        mean_rgb[0, 1] = np.mean(imageRGB[:, 1])
        mean_rgb[0, 2] = np.mean(imageRGB[:, 2])

        mean_lab[0, 0] = np.mean(imageLab[:, 0])
        mean_lab[0, 1] = np.mean(imageLab[:, 1])
        mean_lab[0, 2] = np.mean(imageLab[:, 2])

        mean_hsv[0, 0] = np.mean(imageHVS[:, 0])
        mean_hsv[0, 1] = np.mean(imageHVS[:, 1])
        mean_hsv[0, 2] = np.mean(imageHVS[:, 2])

        self.mean_rgb.append(mean_rgb)
        self.mean_lab.append(mean_lab)
        self.mean_hsv.append(mean_hsv)


    def rgb_diff(self):

        self.red_blue_diff = np.abs(self.mean_rgb[:,0] - self.mean_rgb[:,2])
        self.red_green_diff = np.abs(self.mean_rgb[:, 0] - self.mean_rgb[:, 1])
        self.blue_green_diff = np.abs(self.mean_rgb[:, 2] - self.mean_rgb[:, 1])

    def get_lightness(self):
        for i in range(self.nb_images):
            lab = np.reshape(self.lab[i],(256*256,3))
            self.lightness[i] = np.sum(lab[:,0])

    def get_rgb_lab_hsv_std(self):

        std_rgb = np.zeros((1, 3))
        std_lab = np.zeros((1, 3))
        std_hsv = np.zeros((1, 3))

        imageRGB = np.array(self.rgb[-1])
        imageLab = np.array(self.lab[-1])
        imageHVS = np.array(self.hsv[-1])

        imageRGB = imageRGB.reshape(256 * 256, 3)
        imageLab = imageLab.reshape(256 * 256, 3)
        imageHVS = imageHVS.reshape(256 * 256, 3)

        std_rgb[0, 0] = np.std(imageRGB[:, 0])
        std_rgb[0, 1] = np.std(imageRGB[:, 1])
        std_rgb[0, 2] = np.std(imageRGB[:, 2])

        std_lab[0, 0] = np.std(imageLab[:, 0])
        std_lab[0, 1] = np.std(imageLab[:, 1])
        std_lab[0, 2] = np.std(imageLab[:, 2])

        std_hsv[0, 0] = np.std(imageHVS[:, 0])
        std_hsv[0, 1] = np.std(imageHVS[:, 1])
        std_hsv[0, 2] = np.std(imageHVS[:, 2])

        self.std_rgb.append(std_rgb)
        self.std_lab.append(std_lab)
        self.std_hsv.append(std_hsv)

    def view_histogrammes(self, indexes):
        """
        Affiche les histogrammes de couleur de quelques images
        indexes: int or list of int des images à afficher
        """
        # Pour qu'on puisse traiter 1 seule image
        if type(indexes) == int:
            indexes = [indexes]

        fig = plt.figure()
        ax = fig.subplots(len(indexes), 3)

        for image_counter in range(len(indexes)):
            # charge une image si nécessaire
            if self.all_images_loaded:
                imageRGB = self.images[image_counter]
            else:
                imageRGB = skiio.imread(
                    self.image_folder + os.sep + self.image_list[indexes[image_counter]])

            # Exemple de conversion de format pour Lab et HSV
            imageLab = skic.rgb2lab(imageRGB)  # TODO L1.E4.5: afficher ces nouveaux histogrammes
            imageHSV = skic.rgb2hsv(imageRGB)  # TODO problématique: essayer d'autres espaces de couleur
            # Number of bins per color channel pour les histogrammes (et donc la quantification de niveau autres formats)
            n_bins = 256

            # Lab et HSV requiert un rescaling avant d'histogrammer parce que ce sont des floats au départ!
            imageLabhist = an.rescaleHistLab(imageLab, n_bins)  # External rescale pour Lab
            imageHSVhist = np.round(imageHSV * (n_bins - 1))  # HSV has all values between 0 and 100

            # Construction des histogrammes
            histvaluesRGB = self.generateHistogram(imageRGB)
            histtvaluesLab = self.generateHistogram(imageLabhist)
            histvaluesHSV = self.generateHistogram(imageHSVhist)

            # permet d'omettre les bins très sombres et très saturées aux bouts des histogrammes
            skip = 5
            start = skip
            end = n_bins - skip

            # affichage des histogrammes
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[0, start:end], s=3, c='red')
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[1, start:end], s=3, c='green')
            ax[image_counter, 0].scatter(range(start, end), histvaluesRGB[2, start:end], s=3, c='blue')
            ax[image_counter, 0].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 0].set_title(f'histogramme RGB de {image_name}')

            # 2e histogramme
            # TODO L1.E4 afficher les autres histogrammes de Lab ou HSV dans la 2e colonne de subplots
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[0, start:end], s=3, c="black")
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[1, start:end], s=3, c="red")
            ax[image_counter, 1].scatter(range(start, end), histtvaluesLab[2, start:end], s=3, c="cyan")
            ax[image_counter, 1].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 1].set_title(f'histogramme Lab de {image_name}')

            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[0, start:end], s=3, c="green")
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[1, start:end], s=3, c="red")
            ax[image_counter, 2].scatter(range(start, end), histvaluesHSV[2, start:end], s=3, c="black")
            ax[image_counter, 2].set(xlabel='intensité', ylabel='comptes')
            # ajouter le titre de la photo observée dans le titre de l'histogramme
            image_name = self.image_list[indexes[image_counter]]
            ax[image_counter, 2].set_title(f'histogramme HVS de {image_name}')
