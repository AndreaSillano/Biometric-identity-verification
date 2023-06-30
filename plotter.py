import matplotlib.pyplot as plt


class Plotter:

    def plot_histogram(self, D, L):
        D = D.transpose()

        for i in range(0, D.shape[0]):
            D0 = (D[:, L == 0])[i]
            D1 = (D[:, L == 1])[i]

            plt.hist(D0, bins=100, density=True, ec='black', color="#E23A2E", alpha=0.5, label="Different Speaker")
            plt.hist(D1, bins=100, density=True, ec='black', color="#279847", alpha=0.5, label="Same Speaker")

            plt.legend(loc='upper right')
            # plt.savefig("./images/hist/hist_" + str(i) + ".png")
            plt.show()

    def plot_scatter(self, D, L):
        D = D.transpose()

        for i in range(D.shape[0]):
            for j in range(0, D.shape[0]):
                if j != i:
                    Dx0 = (D[i, L == 0])
                    Dy0 = (D[j, L == 0])
                    plt.scatter(Dx0, Dy0, color="#E23A2E", label="Different Speaker")
                    Dx1 = (D[i, L == 1])
                    Dy1 = (D[j, L == 1])
                    plt.scatter(Dx1, Dy1, color="#279847", label="Same Speaker")
                    plt.legend(loc='upper right')
                    # plt.savefig("./images/scatter/scatter"+str(i)+"_"+str(j)+".png")
                    plt.show()

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