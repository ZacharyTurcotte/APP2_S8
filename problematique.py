"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from helpers.ImageCollection import ImageCollection
from helpers.ClassificationData import ClassificationData
import helpers.analysis as an
import helpers.classifiers as classifiers
import helpers.classifiers
from helpers.ClassificationData import ClassificationData


from keras.optimizers import Adam
import keras as K

import cv2
import helpers.ClassificationData as CD


#######################################
def problematique_APP2():
    print("xd")
    # im_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int) + 100
    # im_list = np.array([0, 1, 4, 2, 8, 9, 3,38,39], dtype=int)
    images = ImageCollection()
    # images = ImageCollection(im_list)

    if True:
        # calculs
        images.count_rgb_pixel()
        images.count_lab_pixel()
        cov = images.get_cov()
        mean_mean = np.mean(images.mean_rgb, axis=1)
        # images.get_edge()
        #     homemade edge detection
        edge_sum = images.edge_detection()
        images.nb_edges = edge_sum

        # normaliser
        mean_mean, minMax_mean_rgm = an.scaleData(mean_mean)
        cov, minMax_cov = an.scaleData(cov)
        images.nb_edges, minMax_nb_edges = an.scaleData(images.nb_edges)

        #  view data 3d
        data_to_view = np.zeros((images.nb_images, 3))
        data_to_view[:, 0] = mean_mean
        data_to_view[:, 1] = cov
        data_to_view[:, 2] = images.nb_edges
        an.view3D(data_to_view, images.target, "3D")

        # separer les classes
        C1 = [];C2 = [];C3 = []
        for i in np.arange(0, images.nb_images):
            if images.target[i] == 0:
                C1.append(np.array([mean_mean[i], cov[i], images.nb_edges[i]]).T)
            if images.target[i] == 1:
                C2.append(np.array([mean_mean[i], cov[i], images.nb_edges[i]]).T)
            if images.target[i] == 2:
                C3.append(np.array([mean_mean[i], cov[i], images.nb_edges[i]]).T)

        # prendre le meme nombre d'images pour chaque classes
        # smallest_class = np.min([len(C1), len(C2), len(C3)])

        smallest_class = 100
        C1 = C1[:smallest_class]
        C2 = C2[:smallest_class]
        C3 = C3[:smallest_class]

        # print stats
        data3classes = ClassificationData([C1, C2, C3])



        ppv1 = classifiers.PPVClassifier(data3classes, n_neighbors=1, metric='minkowski',
                                         useKmean=False, n_represantants=1, experiment_title="1-PPV avec données orig comme représentants",
                                         view=False)

        feature = np.zeros((980, 3))
        for i in np.arange(980):
            feature[i, :] = [mean_mean[i], cov[i], images.nb_edges[i]]
        predictions, errors_indexes = ppv1.predict(feature, images.target, gen_output=True)

        error_rate = np.count_nonzero(images.target - np.resize(predictions, 980)) / 980 * 100



        # ppv5 = classifiers.PPVClassify_APP2(data2train=data3classes, n_neighbors=5,
        #                                     experiment_title='5-PPV avec données orig comme représentants',
        #                                     gen_output=True, view=True)
        # # 1-mean sur chacune des classes
        # # suivi d'un 1-PPV avec ces nouveaux représentants de classes
        # ppv1km1 = classifiers.PPVClassify_APP2(data2train=data3classes, data2test=data3classes, n_neighbors=1,
        #                                        experiment_title='1-PPV sur le 1-moy',
        #                                        useKmean=True, n_representants=9,
        #                                        gen_output=True, view=True)

        # plt.show()

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
        # plt.show()
        print("Done")

    if False:
        n_layers = 6
        n_neurons = [3, 10, 9, 8, 7, 6]

        nn1 = classifiers.NNClassify_APP2(data2train=data_to_view, data2test=data_to_view,
                                          n_layers=n_layers, n_neurons=n_neurons, innerActivation='relu',
                                          outputActivation='softmax', optimizer=Adam(), loss='binary_crossentropy',
                                          metrics=['accuracy'],
                                          callback_list=[K.callbacks.EarlyStopping(patience=50, verbose=1,
                                                                                   restore_best_weights=True),
                                                         classifiers.print_every_N_epochs(25)],
                                          # TODO à compléter L2.E4
                                          experiment_title='NN Simple',
                                          n_epochs=1000, savename='3classes',
                                          ndonnees_random=5000, gen_output=True, view=True)

        print("Done")

    if False:
        print("xd")
        # TODO L1.E4.3 à L1.E4.5
        # Analyser quelques images pour développer des pistes pour le choix de la représentation
        # N = 12
        # im_list = images.get_samples(N)
        # print(im_list)
        # 400,500 -> foret
        # 350 -> plage nuit
        # 100 -> plage jour
    if False:
        im_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + 100

        images.images_display(im_list)
        images.view_histogrammes(im_list)
        plt.show()
        # hist = np.histogram(mean_lab[:,1],bins=128,density=True)
        bins = 128
        n = 2
        range = [-300, 300]
    if False:
        im_list = np.arange(688, 980, 1)
        std_rgb, std_lab, std_hsv, target = images.get_rgb_lab_hsv_std(im_list)
        plt.figure(1)
        plt.hist(std_rgb[:, n], bins=bins, range=range)
        plt.title('street')
        #
        im_list = np.arange(0, 360, 1)
        std_rgb, std_lab, std_hsv, target = images.get_rgb_lab_hsv_std(im_list)
        plt.figure(2)
        plt.hist(std_rgb[:, n], bins=bins, range=range)
        plt.title('coast')

        im_list = np.arange(360, 688, 1)
        std_rgb, std_lab, std_hsv, target = images.get_rgb_lab_hsv_std(im_list)
        plt.figure(3)
        plt.hist(std_rgb[:, n], bins=bins, range=range)
        plt.title('forest')
    if False:
        im_list = np.arange(0, 980, 1)

        data_to_view = np.zeros(((len(im_list)), 3))

        # moy_rgb = np.mean(images.std_rgb,axis=2)
        # moy_lab = np.mean(images.std_lab, axis=2)
        # moy_hsv = np.mean(images.std_hsv,axis=2)

        std_mul_rgb = images.std_rgb[:, 0, 0] + images.std_rgb[:, 0, 1] + images.std_rgb[:, 0, 2]
        std_mul_lab = images.std_lab[:, 0, 0] + images.std_lab[:, 0, 1] + images.std_lab[:, 0, 2]
        std_mul_hsv = images.std_hsv[:, 0, 0] + images.std_hsv[:, 0, 1] + images.std_hsv[:, 0, 2]

        data_to_view[:, 0] = std_mul_rgb
        data_to_view[:, 1] = std_mul_lab
        data_to_view[:, 2] = std_mul_hsv

        an.view3D(data_to_view, images.target, "3D")
        print("Done")

    if 0:
        im_list = im_list + 500
        images.images_display(im_list)
        images.view_histogrammes(im_list)

        # print(CD.)

    # images.generateRepresentation()
    plt.show()


######################################
if __name__ == '__main__':
    problematique_APP2()
