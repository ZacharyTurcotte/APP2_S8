"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""

import matplotlib.pyplot as plt
import numpy as np
from helpers.ImageCollection import ImageCollection
from helpers.ClassificationData import ClassificationData
import helpers.analysis as an
import helpers.classifiers as classifiers
import helpers.classifiers
from helpers.ClassificationData import ClassificationData
import os

from keras.optimizers import Adam
import keras as K
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD
import cv2
import helpers.ClassificationData as CD


#######################################
def problematique_APP2():
    print("xd")
    # im_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int) + 100
    # im_list = np.array([0, 1, 4, 2, 8, 9, 3,38,39], dtype=int)
    images = ImageCollection()
    # images = ImageCollection(im_list)

    #im_list = np.array([100, 110, 450, 500, 600, 700, 800, 900, 950, 50], dtype=int) + 0
    im_list = np.arange(0,980,5)
    images = ImageCollection(load_img=None)
    print("d")

    # 0 = kmean
    # 1 = NN
    # 2 = Bayes
    choix_classificateur = 2

    if True:
        # calculs
        images.count_rgb_pixel()
        images.count_lab_pixel()
        cov = images.get_cov()
        mean_mean = np.mean(images.mean_rgb, axis=1)
        images.get_edge()
        # homemade edge detection
        # edge_sum = images.edge_detection()
        # images.nb_edges = edge_sum

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
        dims = [mean_mean,cov,images.nb_edges]

        [C1_train,C2_train,C3_train,C1_test,C2_test,C3_test] = images.split_data_PPV(200,dims)

        images.nb_grey_pixels

        min_len = np.min([len(C1_test),len(C2_test),len(C3_test)])

        # print stats
        data3classes_train = ClassificationData([C1_train, C2_train, C3_train])
        data3classes_test = ClassificationData([C1_test[:min_len,:], C2_test[:min_len,:], C3_test[:min_len,:]])

        ######################################################################
        # Kmeans, 1 PPV
        ######################################################################
        if choix_classificateur == 0:
            ppv1 = classifiers.PPVClassifier(data3classes_train, n_neighbors=3, metric='minkowski',
                                             useKmean=True, n_represantants=9, experiment_title="1-PPV avec données orig comme représentants",
                                             view=False)


            predictions, errors_indexes = ppv1.predict(data3classes_test.data1array, data3classes_test.labels1array, gen_output=True)

            error_rate = np.count_nonzero(images.target - np.resize(predictions, 980)) / 980 * 100
            print(error_rate)
            plt.show()

        ######################################################################
        # Reseau de neuron
        ######################################################################
        if (choix_classificateur == 1):
            n_neurons = [4,10,9,8,7,6]
            n_layers = 6
            # shuffledTrainData, shuffledTrainLabels, shuffledValidData, shuffledValidLabels = an.splitDataNN(3, data3classes, target)
            # data3classes

            model = Sequential()
            model.add(Dense(units=3, activation='tanh',
                            input_shape=(len(dims),)))
            model.add(Dense(units=10, activation='tanh',
                            ))
            model.add(Dense(units=9, activation='tanh',
                            ))
            model.add(Dense(units=8, activation='tanh',
                            ))
            model.add(Dense(units=7, activation='tanh',
                            ))
            model.add(Dense(units=6, activation='tanh',
                            ))
            # model.add(Dense(units=40, activation='tanh',
            #                input_shape=(data.shape[-1],)))
            model.add(Dense(units=3, activation='sigmoid'))  #

            print(model.summary())

            model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.01), loss='mse')
            print("compile done")
            model.fit(data3classes_train.data1array, data3classes_train.labels1array, batch_size=10, verbose=1,
                      epochs=1000, shuffle=False)  # TODO Labo: ajouter les arguments pour le validation set
            print("fit done")
            # Save trained model to disk
            model.save('saves' + os.sep + 'iris.keras')
            print("save done")
            an.plot_metrics(model)

            # Test model (loading from disk)
            model = load_model('saves' + os.sep + 'iris.keras')
            targetPred = model.predict(data3classes_test.data1array)
            print("predict done")
            # Print the number of classification errors from the training data
            error_indexes = an.calc_erreur_classification(np.argmax(targetPred, axis=-1), data3classes_test.labels1array, gen_output=True)

            plt.show()

        ######################################################################
        # Bayes
        ######################################################################
        if choix_classificateur == 2:
            ## bayes

            apriori = [1/3,1/3,1/3]
            costs = [[0, 1, 1], [1, 0, 1], [1, 1, 0]] # le cost nous permet d'avoir du control sur les frontières.
            # Bayes gaussien les apriori et coûts ne sont pas considérés pour l'instant
            bg1 = classifiers.BayesClassifier(data3classes_train, classifiers.GaussianProbDensity, apriori=apriori, costs=costs)

            predictions, errors_indexes = bg1.predict(data3classes_test.data1array, data3classes_test.labels1array, gen_output=True)

            error_rate = np.count_nonzero(images.target - np.resize(predictions, 980)) / 980 * 100
            print(error_rate)
            plt.show()
            plt.show()


        print("Done")

    if False:
        cov = images.get_cov()

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
