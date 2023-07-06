from mlFunc import *
class GMM:

    def logpdf_GMM(self,D, gmm):
        S = numpy.zeros((len(gmm), D.shape[1]))

        for g in range(len(gmm)):
            # print (w,mu,C)
            (w, mu, C) = gmm[g]
            S[g, :] = self.logpdf_GAU_ND(D, mu, C) + numpy.log(w)

        logdens = scipy.special.logsumexp(S, axis=0)

        return S, logdens

    def logpdf_GAU_ND(self, D, mu, C):

        res = -0.5 * D.shape[0] * numpy.log(2 * numpy.pi)
        res += -0.5 * numpy.linalg.slogdet(C)[1]
        res += -0.5 * ((D - mu) * numpy.dot(numpy.linalg.inv(C), (D - mu))).sum(0)  # 1
        return res

    def LBG_FULL(self,DTR, alpha,components, psi):
        U, s, _ = numpy.linalg.svd(empirical_covariance(DTR, empirical_mean(DTR)))
        s[s < psi] = psi
        covNew = numpy.dot(U, vcol(s) * U.T)
        GMM = [(1, empirical_mean(DTR), covNew)]

        while len(GMM) <= components:
            if len(GMM) != 1:
                GMM = self.GMM_EM(DTR, GMM, psi)
            if len(GMM) == components:
                break

            gmmNew = []
            for i in range(len(GMM)):
                # nuove componenti
                (w, mu, sigma) = GMM[i]
                U, s, vh = numpy.linalg.svd(sigma)
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                gmmNew.append((w / 2, mu + d, sigma))
                gmmNew.append((w / 2, mu - d, sigma))
            GMM = gmmNew

        return GMM

    def LBG_NAIVE(self,DTR, alpha,components, psi):
        U, s, _ = numpy.linalg.svd(empirical_covariance(DTR, empirical_mean(DTR)))
        s[s < psi] = psi
        covNew = numpy.dot(U, vcol(s) * U.T)
        GMM = [(1, empirical_mean(DTR), covNew)]

        while len(GMM) <= components:
            if len(GMM) != 1:
                GMM = self.GMM_EM_NAIVE(DTR, GMM, psi)
            if len(GMM) == components:
                break

            gmmNew = []
            for i in range(len(GMM)):
                # nuove componenti
                (w, mu, sigma) = GMM[i]
                U, s, vh = numpy.linalg.svd(sigma)
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                gmmNew.append((w / 2, mu + d, sigma))
                gmmNew.append((w / 2, mu - d, sigma))
            GMM = gmmNew

        return GMM

    def LBG_TIED(self,DTR, alpha,components, psi):
        U, s, _ = numpy.linalg.svd(empirical_covariance(DTR, empirical_mean(DTR)))
        s[s < psi] = psi
        covNew = numpy.dot(U, vcol(s) * U.T)
        GMM = [(1, empirical_mean(DTR), covNew)]

        while len(GMM) <= components:
            if len(GMM) != 1:
                GMM = self.GMM_EM_TIED(DTR, GMM, psi)
            if len(GMM) == components:
                break

            gmmNew = []
            for i in range(len(GMM)):
                # nuove componenti
                (w, mu, sigma) = GMM[i]
                U, s, vh = numpy.linalg.svd(sigma)
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                gmmNew.append((w / 2, mu + d, sigma))
                gmmNew.append((w / 2, mu - d, sigma))
            GMM = gmmNew

        return GMM

    def LBG_TIEDNAIVE(self,DTR, alpha,components, psi):
        U, s, _ = numpy.linalg.svd(empirical_covariance(DTR, empirical_mean(DTR)))
        s[s < psi] = psi
        covNew = numpy.dot(U, vcol(s) * U.T)
        GMM = [(1, empirical_mean(DTR), covNew)]

        while len(GMM) <= components:
            if len(GMM) != 1:
                GMM = self.GMM_EM_TIEDNAIVE(DTR, GMM, psi)
            if len(GMM) == components:
                break

            gmmNew = []
            for i in range(len(GMM)):
                # nuove componenti
                (w, mu, sigma) = GMM[i]
                U, s, vh = numpy.linalg.svd(sigma)
                d = U[:, 0:1] * s[0] ** 0.5 * alpha
                gmmNew.append((w / 2, mu + d, sigma))
                gmmNew.append((w / 2, mu - d, sigma))
            GMM = gmmNew

        return GMM

    def predict_GMM_full(self, DTR, LTR,Dte, componentsT, componentsNT, a, p):
        D0 = DTR[:, LTR == 0]
        D1 = DTR[:, LTR == 1]
        gmm0 = self.LBG_FULL(D0, a, componentsNT, p)
        _, llr0 = self.logpdf_GMM(Dte, gmm0)

        gmm1 = self.LBG_FULL(D1, a, componentsT, p)
        _, llr1 = self.logpdf_GMM(Dte, gmm1)

        return llr1-llr0


    def GMM_EM(self, D, gmm, psi=0.01):
        '''
        EM algorithm for GMM full covariance
        It estimates the parameters of a GMM that maximize the ll for
        a training set X
        If psi is given it's used for constraining the eigenvalues of the
        covariance matrices to be larger or equal to psi
        '''
        llNew = None
        llOld = None
        G = len(gmm)
        N = D.shape[1]

        while llOld is None or llNew - llOld > 1e-6:
            llOld = llNew
            SJ, SM = self.logpdf_GMM(D, gmm)
            llNew = SM.sum() / N
            P = numpy.exp(SJ - SM)
            gmmNew = []
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (vrow(gamma) * D).sum(1)
                S = numpy.dot(D, (vrow(gamma) * D).T)
                w = Z / N
                mu = vcol(F / Z)
                Sigma = S / Z - numpy.dot(mu, mu.T)
                U, s, _ = numpy.linalg.svd(Sigma)
                s[s < psi] = psi
                Sigma = numpy.dot(U, vcol(s) * U.T)
                gmmNew.append((w, mu, Sigma))
            gmm = gmmNew
        return gmm

    def predict_GMM_naive(self, DTR, LTR, Dte, componentsT, componentsNT, a, p):
        D0 = DTR[:, LTR == 0]
        D1 = DTR[:, LTR == 1]
        gmm0 = self.LBG_NAIVE(D0, a, componentsNT, p)
        _, llr0 = self.logpdf_GMM(Dte, gmm0)

        gmm1 = self.LBG_NAIVE(D1, a, componentsT, p)
        _, llr1 = self.logpdf_GMM(Dte, gmm1)

        return llr1 - llr0

    def GMM_EM_NAIVE(self,D, gmm, psi=0.01):
        '''
            EM algorithm for GMM diagonal covariance
            It estimates the parameters of a GMM that maximize the ll for
            a training set X
            If psi is given it's used for constraining the eigenvalues of the
            covariance matrices to be larger or equal to psi
            '''
        llNew = None
        llOld = None
        G = len(gmm)
        N = D.shape[1]
        while llOld is None or llNew - llOld > 1e-6:
            llOld = llNew
            SJ, SM = self.logpdf_GMM(D, gmm)
            llNew = SM.sum() / N
            P = numpy.exp(SJ - SM)

            gmmNew = []
            for g in range(G):
                # m step
                gamma = P[g, :]
                Z = gamma.sum()
                F = (vrow(gamma) * D).sum(1)
                S = numpy.dot(D, (vrow(gamma) * D).T)
                w = Z / N
                mu = vcol(F / Z)
                Sigma = S / Z - numpy.dot(mu, mu.T)
                Sigma = Sigma * numpy.eye(Sigma.shape[0])
                U, s, _ = numpy.linalg.svd(Sigma)
                s[s < psi] = psi
                sigma = numpy.dot(U, vcol(s) * U.T)
                gmmNew.append((w, mu, sigma))
            gmm = gmmNew
        return gmm

    def predict_GMM_TiedCov(self, DTR, LTR, Dte, componentsT, componentsNT, a, p):
        D0 = DTR[:, LTR == 0]
        D1 = DTR[:, LTR == 1]
        gmm0 = self.LBG_TIED(D0, a, componentsNT, p)
        _, llr0 = self.logpdf_GMM(Dte, gmm0)

        gmm1 = self.LBG_TIED(D1, a, componentsT, p)
        _, llr1 = self.logpdf_GMM(Dte, gmm1)

        return llr1 - llr0

    def GMM_EM_TIED(self, D, gmm, psi=0.01):
        '''
        EM algorithm for GMM tied full covariance
        It estimates the parameters of a GMM that maximize the ll for
        a training set X
        If psi is given it's used for constraining the eigenvalues of the
        covariance matrices to be larger or equal to psi
        '''
        llNew = None
        llOld = None
        G = len(gmm)
        N = D.shape[1]
        sigma_array = []
        while llOld is None or llNew - llOld > 1e-6:
            llOld = llNew
            SJ, SM = self.logpdf_GMM(D, gmm)
            llNew = SM.sum() / N
            P = numpy.exp(SJ - SM)
            gmmNew = []

            sigmaTied = numpy.zeros((D.shape[0], D.shape[0]))
            for g in range(G):
                # m step
                gamma = P[g, :]
                Z = gamma.sum()
                F = (vrow(gamma) * D).sum(1)
                S = numpy.dot(D, (vrow(gamma) * D).T)
                w = Z / N
                mu = vcol(F / Z)
                Sigma = S / Z - numpy.dot(mu, mu.T)
                sigmaTied += Z * Sigma
                gmmNew.append((w, mu))

            # calculate tied covariance
            gmm = gmmNew
            sigmaTied /= N
            U, s, _ = numpy.linalg.svd(sigmaTied)
            s[s < psi] = psi
            sigmaTied = numpy.dot(U, vcol(s) * U.T)

            gmmNew = []
            for g in range(G):
                (w, mu) = gmm[g]
                gmmNew.append((w, mu, sigmaTied))
            gmm = gmmNew
        return gmm

    def predict_GMM_TiedNaive(self, DTR, LTR, Dte, componentsT, componentsNT, a, p):
        D0 = DTR[:, LTR == 0]
        D1 = DTR[:, LTR == 1]
        gmm0 = self.LBG_TIEDNAIVE(D0, a, componentsNT, p)
        _, llr0 = self.logpdf_GMM(Dte, gmm0)

        gmm1 = self.LBG_TIEDNAIVE(D1, a, componentsT, p)
        _, llr1 = self.logpdf_GMM(Dte, gmm1)

        return llr1 - llr0

    def GMM_EM_TIEDNAIVE(self,D, gmm, psi=0.01):  # X -> ev
        '''
        EM algorithm for GMM tied diagonal covariance
        It estimates the parameters of a GMM that maximize the ll for
        a training set X
        If psi is given it's used for constraining the eigenvalues of the
        covariance matrices to be larger or equal to psi
        '''
        llNew = None
        llOld = None
        G = len(gmm)
        N = D.shape[1]
        sigma_array = []
        while llOld is None or llNew - llOld > 1e-6:
            llOld = llNew
            SJ = numpy.zeros((G, N))
            for g in range(G):
                SJ[g, :] = self.logpdf_GAU_ND(
                    D, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
            SM = scipy.special.logsumexp(SJ, axis=0)
            llNew = SM.sum() / N
            P = numpy.exp(SJ - SM)
            gmmNew = []
            sigmaTied = numpy.zeros((D.shape[0], D.shape[0]))
            for g in range(G):
                # m step
                gamma = P[g, :]
                Z = gamma.sum()
                F = (vrow(gamma) * D).sum(1)
                S = numpy.dot(D, (vrow(gamma) * D).T)
                w = Z / N
                mu = vcol(F / Z)
                # Sigma = S/Z - numpy.dot(mu, mu.T)
                # Sigma = Sigma * numpy.eye(Sigma.shape[0])
                # Sigma = Sigma * Z
                # gmmNew.append((w, mu, Sigma))
                sigma = S / Z - numpy.dot(mu, mu.T)
                sigmaTied += Z * sigma
                gmmNew.append((w, mu))

                # calculate tied covariance
            gmm = gmmNew
            sigmaTied /= N
            sigmaTied *= numpy.eye(sigma.shape[0])
            U, s, _ = numpy.linalg.svd(sigmaTied)
            s[s < psi] = psi
            sigmaTied = numpy.dot(U, vcol(s) * U.T)

            newGmm = []
            for i in range(len(gmm)):
                (w, mu) = gmm[i]
                newGmm.append((w, mu, sigmaTied))

            gmm = newGmm
            # print(llNew,'llnew') #increase - if decrease problem
        # print(llNew-llOld,'llnew-llold')
        return gmm