import matplotlib.pyplot as plt
from mlFunc import *
import numpy as numpy

class Plotter:

    def plot_bar_GMM(self, data):
        K_T = ["1","2", "4"]
        x = numpy.arange(len(K_T))
        width = 0.10
        multiplier = 0
        print(data)
        for K, val in data.items():
            offset = width * multiplier
            rects = plt.bar(x + offset, val, width, label=K)
            plt.bar_label(rects, padding=3)
            multiplier += 1
        plt.ylabel('minDCF')
        plt.xlabel('K-Target')
        plt.xticks(x + width, K_T)
        plt.legend(loc='upper left')

        plt.show()

    def plot_histogram(self, D, L):
        D = D.transpose()

        for i in range(0, D.shape[0]):
            D0 = (D[:, L == 0])[i]
            D1 = (D[:, L == 1])[i]

            plt.hist(D0, bins=100, density=True, ec='black', color="#E23A2E", alpha=0.5, label="Spoofed Fingerprint")
            plt.hist(D1, bins=100, density=True, ec='black', color="#279847", alpha=0.5, label="Autenthic Fingerprint")

            plt.legend(loc='upper right')
            #plt.savefig("./images/hist/hist_" + str(i) + ".png")
            plt.show()
            #plt.close()
            
    def plot_scatter(self, D, L):
        D = D.transpose()

        for i in range(D.shape[0]):
            for j in range(0, D.shape[0]):
                if j != i:
                    Dx0 = (D[i, L == 0])
                    Dy0 = (D[j, L == 0])
                    plt.scatter(Dx0, Dy0, color="#E23A2E",  label="Spoofed Fingerprint")
                    Dx1 = (D[i, L == 1])
                    Dy1 = (D[j, L == 1])
                    plt.scatter(Dx1, Dy1, color="#279847",label="Autenthic Fingerprint")
                    plt.legend(loc='upper right')
                    plt.savefig("./images/scatter/scatter"+str(i)+"_"+str(j)+".png")
                    plt.close()

                    # plt.show()
    def plot_PCA_scatter(self, D, L):
        Dx0 = (D[0, L == 0])
        Dy0 = (D[1, L == 0])
        plt.scatter(Dx0, Dy0, color="#E23A2E", label="Different Speaker")
        Dx1 = (D[0, L == 1])
        Dy1 = (D[1, L == 1])
        plt.scatter(Dx1, Dy1, color="#279847", label="Same Speaker")
        plt.legend(loc='upper right')

        plt.show()

    def plot_LDA_scatter(self, D, L):
        Dx0 = (D[0, L == 0])
        Dy0 = (D[1, L == 0])
        plt.scatter(Dx0, Dy0, color="#E23A2E", label="Different Speaker")
        Dx1 = (D[0, L == 1])
        Dy1 = (D[1, L == 1])
        plt.scatter(Dx1, Dy1, color="#279847", label="Same Speaker")
        plt.legend(loc='upper right')

        plt.show()

    def plot_correlations(self, DTR, title, cmap="Greys"):
        corr = numpy.zeros((10, 10))
        for x in range(10):
            for y in range(10):
                X = DTR[x, :]
                Y = DTR[y, :]
                pearson_elem = compute_correlation(X, Y)
                corr[x][y] = pearson_elem

        plt.rcParams['axes.linewidth'] = 0.2

        fig, ax = plt.subplots()
        heatmap = ax.imshow(numpy.abs(corr), cmap=cmap, aspect='equal')

        ax.set_xticks(numpy.arange(corr.shape[1]))
        ax.set_yticks(numpy.arange(corr.shape[0]))
        ax.set_xticklabels(numpy.arange(corr.shape[1]))
        ax.set_yticklabels(numpy.arange(corr.shape[0]))
        ax.tick_params(axis='both', which='both', length=0)

        cbar = plt.colorbar(heatmap)

        plt.show()
        fig = heatmap.get_figure()
        fig.savefig("./images/" + title + ".png")

    def plot_DCF_lambda(self, x, y_05, y_01, y_09, xlabel, title=''):
        plt.figure()
        plt.plot(x, y_05, label='min DCF prior=0.5', color='b')
        plt.plot(x, y_09, label='min DCF prior=0.9', color='g')
        plt.plot(x, y_01, label='min DCF prior=0.1', color='r')
        plt.xlim([min(x), max(x)])
        plt.xscale("log", base=10)
        plt.legend(loc='upper left')
        plt.xlabel(xlabel)
        plt.ylabel("min DCF")
        plt.savefig('./images/DCF_' + title + '.png')
        plt.show()
        plt.close()

    def plot_DCF_SVM_C(self, x, y_lin, y_pol, y_rbf, xlabel, title=''):
        plt.figure()
        plt.plot(x, y_lin, label='SVM Polynomial', color='b')
        plt.plot(x, y_pol, label='SVM Polynomial PCA 7', color='g')
        plt.plot(x, y_rbf, label='SVM Polynomial (Z-norm)', color='r')
        plt.xlim([min(x), max(x)])
        plt.xscale("log", base=10)
        plt.legend(loc='upper left')
        plt.xlabel(xlabel)
        plt.ylabel("min DCF")
        plt.savefig('./images/DCF_' + title + '.png')
        plt.show()

    def plot_DCF_compare(self, x, y, y_z):
        plt.figure()
        plt.plot(x, y_z, label='Log-Reg (z-norm)', color='b')
        plt.plot(x, y, label='Log-Reg', color='r')
        plt.xlim([min(x), max(x)])
        plt.xscale("log", base=10)
        plt.legend(loc='upper left')
        plt.xlabel('lambda')
        plt.ylabel("min DCF")
        #plt.savefig('./images/DCF_' + 'LR' + '.png')
        plt.show()
    def plot_DCF_compare_QUAD(self, x, y, y_z):
        plt.figure()
        plt.plot(x, y_z, label='Q-Log-Reg (z-norm)', color='b')
        plt.plot(x, y, label='Q-Log-Reg', color='r')
        plt.xlim([min(x), max(x)])
        plt.xscale("log", base=10)
        plt.legend(loc='upper left')
        plt.xlabel('lambda')
        plt.ylabel("min DCF")
        #plt.savefig('./images/DCF_' + 'LR' + '.png')
        plt.show()

    def plot_DCF_compare_PCA(self, x, y, y_9, y_8, y_7):
        plt.figure()
        plt.plot(x, y, label='Log-Reg', color='r' ,linestyle='dashed')
        plt.plot(x, y_8, label='Log-Reg PCA-8', color='b')
        plt.plot(x, y_7, label='Log-Reg PCA-7', color='y')
        plt.plot(x, y_9, label='Log-Reg PCA-9', color='g')
        plt.xlim([min(x), max(x)])
        plt.xscale("log", base=10)
        plt.legend(loc='upper left')
        plt.xlabel('lambda')
        plt.ylabel("min DCF")
        #plt.savefig('./images/DCF_' + title + '.svg')
        plt.show()

    def plot_DCF_compare_PCA_SVM(self, x, y, y_9, y_8, y_7, title=''):
        plt.figure()
        plt.plot(x, y, label='SVM Linear', color='r' ,linestyle='dashed')
        plt.plot(x, y_8, label='SVM Linear PCA-8', color='b')
        plt.plot(x, y_7, label='SVM Linear (z-norm)', color='y')
        plt.plot(x, y_9, label='SVM Linear PCA-9', color='g')
        plt.xlim([min(x), max(x)])
        plt.xscale("log", base=10)
        plt.legend(loc='upper left')
        plt.xlabel('C')
        plt.ylabel("min DCF")
        plt.savefig('./images/DCF_' + title + '.svg')
        plt.show()
        plt.close()
    def plot_DCF_compare_PCA_Q(self, x, y, y_9, y_8, y_7, y_6):
        plt.figure()
        plt.plot(x, y, label='Log-Reg', color='r' ,linestyle='dashed')
        plt.plot(x, y_8, label='Q-Log-Reg PCA-8', color='b')
        plt.plot(x, y_7, label='Q-Log-Reg PCA-7', color='y')
        plt.plot(x, y_9, label='Q-Log-Reg PCA-9', color='g')
        plt.plot(x, y_6, label='Q-Log-Reg PCA-6', color='c')
        plt.xlim([min(x), max(x)])
        plt.xscale("log", base=10)
        plt.legend(loc='upper left')
        plt.xlabel('lambda')
        plt.ylabel("min DCF")
        #plt.savefig('./images/DCF_' + title + '.svg')
        plt.show()
    def plot_DCF_compare_PCA_Z(self, x, y, y_z):
        plt.figure()
        plt.plot(x, y_z, label='Q-Log-Reg PCA-7 (z-norm)', color='b')
        plt.plot(x, y, label='Q-Log-Reg PCA-7', color='r')
        plt.xlim([min(x), max(x)])
        plt.xscale("log", base=10)
        plt.legend(loc='upper left')
        plt.xlabel('lambda')
        plt.ylabel("min DCF")
        # plt.savefig('./images/DCF_' + 'LR' + '.png')
        plt.show()

    def ROC_curve(self, firstModel, secondModel, thirdModel, fourthModel, LTR1, LTR2, LTR3, LTR4):
        plt.figure()

        thresholds = numpy.array(firstModel)
        thresholds.sort()
        thresholds = numpy.ravel(thresholds)
        thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
        FPR = numpy.zeros(thresholds.size)
        TPR = numpy.zeros(thresholds.size)
        for idx, t in enumerate(thresholds):
            pred_label = numpy.int32(firstModel > t)
            conf = confusion_matrix_binary(pred_label, LTR1)
            TPR[idx] = conf[1, 1] / (conf[1, 1] + conf[0, 1])
            FPR[idx] = conf[1, 0] / (conf[1, 0] + conf[0, 0])
        plt.plot(FPR, TPR, label='MVG Full PCA 8', color='r')

        thresholds = numpy.array(secondModel)
        thresholds.sort()
        thresholds = numpy.ravel(thresholds)
        thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
        FPR = numpy.zeros(thresholds.size)
        TPR = numpy.zeros(thresholds.size)
        for idx, t in enumerate(thresholds):
            pred_label = numpy.int32(firstModel > t)
            conf = confusion_matrix_binary(pred_label, LTR2)
            TPR[idx] = conf[1, 1] / (conf[1, 1] + conf[0, 1])
            FPR[idx] = conf[1, 0] / (conf[1, 0] + conf[0, 0])
        plt.plot(FPR, TPR, label='Quadratic LogReg PCA 7', color='b')

        thresholds = numpy.array(thirdModel)
        thresholds.sort()
        thresholds = numpy.ravel(thresholds)
        thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
        FPR = numpy.zeros(thresholds.size)
        TPR = numpy.zeros(thresholds.size)
        for idx, t in enumerate(thresholds):
            pred_label = numpy.int32(thirdModel > t)
            conf = confusion_matrix_binary(pred_label, LTR3)
            TPR[idx] = conf[1, 1] / (conf[1, 1] + conf[0, 1])
            FPR[idx] = conf[1, 0] / (conf[1, 0] + conf[0, 0])
        plt.plot(FPR, TPR, label='RBF SVM', color='y')

        thresholds = numpy.array(fourthModel)
        thresholds.sort()
        thresholds = numpy.ravel(thresholds)
        thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
        FPR = numpy.zeros(thresholds.size)
        TPR = numpy.zeros(thresholds.size)
        for t in enumerate(thresholds):
            pred_label = numpy.int32(thirdModel > t)
            conf = confusion_matrix_binary(pred_label, LTR4)
            TPR[idx] = conf[1, 1] / (conf[1, 1] + conf[0, 1])
            FPR[idx] = conf[1, 0] / (conf[1, 0] + conf[0, 0])
        plt.plot(FPR, TPR, label='GMM Naive 1-8', color='g')

        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.savefig('images/comparison/' + 'ROC.png')
        plt.show()
