"""
Script de départ de la problématique
Problématique APP2 Module IA S8
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
from helpers.ImageCollection import ImageCollection
from helpers.ClassificationData import ClassificationData
import helpers.analysis as an
import helpers.classifiers as classifiers
import helpers.classifiers
from helpers.ClassificationData import ClassificationData
from sklearn.preprocessing import OneHotEncoder
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

    # im_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int) + 100
    # im_list = np.array([0, 1, 4, 2, 8, 9, 3,38,39], dtype=int)
    #images = ImageCollection()
    # images = ImageCollection(im_list)

    #im_list = np.array([100, 110, 450, 500, 600, 700, 800, 900, 950, 50], dtype=int) + 0
    im_list = np.arange(0,980,5)
    images = ImageCollection(load_img=None) # on peut ajouter une liste d'index pour loader des images en particulier et non toutes
    print("Done loading images")
    # 0 = k mean
    # 1 = NN
    # 2 = Bayes
    choix_classificateur = 0 #endroit ou change quel classificateur on veut
    grey_pixel = images.count_grey_pixel()
    if True:
        # calculs
        if True:
            images.count_rgb_pixel()
            images.count_lab_pixel()
            cov = images.get_cov()
            mean_mean = np.mean(images.mean_rgb, axis=1)
            images.get_edge()
            #edges = images.edge_detection(view=False)
            images.get_lightness()
            # homemade edge detection
            # edge_sum = images.edge_detection()
            # images.nb_edges = edge_sum

            # normaliser
            mean_mean, minMax_mean_rgm = an.scaleData(mean_mean)
            cov, minMax_cov = an.scaleData(cov)
            edges, minMax_nb_edges = an.scaleData(images.nb_edges)
            #edges,_ = an.scaleData(edges)
            grey_pixel_norm,_ = an.scaleData(grey_pixel)
            #green_pixel_norm,_ = an.scaleData(images.nb_green_pixels_lab)
            #lightness,_ = an.scaleData(images.lightness)
            #  view data 3d
            data_to_view = np.zeros((images.nb_images, 3))
            data_to_view[:, 0] = mean_mean
            data_to_view[:, 1] = cov
            data_to_view[:, 2] = edges
            an.view3D(data_to_view, images.target, "3D")
            #plt.show()
            dims = [mean_mean,cov,edges]

            #with open('my_arrays.pkl', 'wb') as file:
             #   pickle.dump(dims, file)
        print("allo")
        #with open('my_arrays.pkl', 'rb') as file:
        #    dims = pickle.load(file)

        [C1_train,C2_train,C3_train,C1_test,C2_test,C3_test] = images.split_data(250,dims)
        #split les données également car ClassificationData ne fonctionne pas autrement.

        #images.nb_grey_pixels

        min_len = np.min([len(C1_test),len(C2_test),len(C3_test)])

        # print stats
        data3classes_train = ClassificationData([C1_train, C2_train, C3_train])
        data3classes_test = ClassificationData([C1_test[:min_len,:], C2_test[:min_len,:], C3_test[:min_len,:]])

        #data3classes_test_label_encode = OneHotEncoder(sparse_output=False).fit_transform(data3classes_test.labels1array)

        #data3classes_train_label_encode = OneHotEncoder(sparse_output=False).fit_transform(data3classes_train.labels1array)
        ######################################################################
        # Kmeans, 1 PPV
        ######################################################################
        if choix_classificateur == 0:

            ppv1 = classifiers.PPVClassifier(data3classes_train, n_neighbors=3, metric='minkowski', #mettre n=2 donc euclide
                                             useKmean=True, n_represantants=3, experiment_title="1-PPV avec données orig comme représentants",
                                             view=True)



        # def __init__(self, data2train, data2test, n_layers, n_neurons, innerActivation='tanh',
        #              outputActivation='softmax',
        #              optimizer=Adam(), loss='binary_crossentropy', metrics=None,
        #              callback_list=None, n_epochs=1000, savename='',
        # ppv1 = classifiers.PPVClassifier(data3classes_train, n_neighbors=3, metric='minkowski',
        #                                  useKmean=True, n_represantants=9, experiment_title="1-PPV avec données orig comme représentants",
        #                                  view=False)
        #
        #
        # predictions, errors_indexes = ppv1.predict(data3classes_test.data1array, data3classes_test.labels1array, gen_output=True)
            #predictTest = []
            #error_indexes = []

            predictions, errors_indexes = ppv1.predict(data3classes_test.data1array, data3classes_test.labels1array, gen_output=True)

            random_data = np.random.uniform(-1, 1, size=(5000,3))
            predictions_random,errors_indexes = ppv1.predict(random_data, gen_output=True)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(random_data[:,0],random_data[:,1],random_data[:,2],c=predictions_random)
            ax.set_xlabel('Couleur moyenne')
            ax.set_ylabel('Covariance')
            ax.set_zlabel('Nombre de limites (edges)')

            error_rate = np.count_nonzero(images.target - np.resize(predictions, 980)) / 980 * 100
            print(error_rate)
            plt.show()
            print("Done")



        ######################################################################
        # Reseau de neuron
        ######################################################################
        if (choix_classificateur == 1):
            NN = classifiers.NNClassify_prob(data3classes_train, data3classes_test, 6,
                                             [3, 10, 10, 10, 10, 10, 10, 10, 8 , 8, 10,10,6,6,6,6,6,6,6,6,6], #nombre de neurone par couches
                                             gen_output=True,n_epochs=2500,metrics=['accuracy'],
                                             callback_list=[K.callbacks.EarlyStopping(patience=50,verbose=1,restore_best_weights=True),
                                                         classifiers.print_every_N_epochs(25)])

            plt.show()
            print("D0ne")

        ######################################################################
        # Bayes
        ######################################################################
        if choix_classificateur == 2:
            ## bayes
            random_data1 = np.random.uniform(-0.5, 0.5, size=(5000, 1))
            random_data2 = np.random.uniform(-0.75,0.25,size=(5000,1))
            random_data3 = np.random.uniform(-1,1,size=(5000,1))
            random_data = np.random.uniform(-0.5,0.5,size=(5000,3))
            #random_data = np.concatenate((random_data1,random_data2,random_data3),axis=1).reshape((5000,3))

            apriori =  [1/3,1/3,1/3]
            costs = [[0, 1, 1], [2, 0, 1], [2, 1, 0]] # le cost nous permet d'avoir du control sur les frontières.
            # Bayes gaussien les apriori et coûts ne sont pas considérés pour l'instant
            bg1 = classifiers.BayesClassifier(data3classes_train, classifiers.HistProbDensity, nb_bins=10,apriori=apriori, costs=costs)

            bg1.predict(data3classes_test.data1array, data3classes_test.labels1array, gen_output=True)
            prediction_random, error_random = bg1.predict(random_data)

            mini = -1
            maxi = 1

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(random_data[:,0],random_data[:, 1], random_data[:, 2], c=prediction_random)
            ax.set_xlabel('Couleur moyenne')
            ax.set_ylabel('Covariance')
            ax.set_zlabel('Nombre de limites (edges)')
            # ax.set_xlim(mini, maxi)
            # ax.set_ylim(mini, maxi)
            # ax.set_zlim(mini, maxi)
            plt.show()
            print("Done")
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


######################################
if __name__ == '__main__':
    problematique_APP2()
