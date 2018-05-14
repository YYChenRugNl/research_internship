# -*- coding: utf-8 -*-

# Author: Joris Jensen <jjensen@techfak.uni-bielefeld.de>
#
# License: BSD 3 clause

from __future__ import division

import math
from math import log

import numpy as np
from scipy.optimize import minimize
import random

from .glvq import GlvqModel, _squared_euclidean
from sklearn.utils import validation


class OGmlvqModel(GlvqModel):

    """Generalized Matrix Learning Vector Quantization

    Parameters
    ----------

    prototypes_per_class : int or list of int, optional (default=1)
        Number of prototypes per class. Use list to specify different numbers
        per class.

    initial_prototypes : array-like,
     shape =  [n_prototypes, n_features + 1], optional
        Prototypes to start with. If not given initialization near the class
        means. Class label must be placed as last entry of each prototype

    initial_matrix : array-like, shape = [dim, n_features], optional
        Relevance matrix to start with.
        If not given random initialization for rectangular matrix and unity
        for squared matrix.

    regularization : float, optional (default=0.0)
        Value between 0 and 1. Regularization is done by the log determinant
        of the relevance matrix. Without regularization relevances may
        degenerate to zero.

    dim : int, optional (default=nb_features)
        Maximum rank or projection dimensions

    max_iter : int, optional (default=2500)
        The maximum number of iterations.

    gtol : float, optional (default=1e-5)
        Gradient norm must be less than gtol before successful
        termination of l-bfgs-b.

    display : boolean, optional (default=False)
        Print information about the bfgs steps.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------

    w_ : array-like, shape = [n_prototypes, n_features]
        Prototype vector, where n_prototypes in the number of prototypes and
        n_features is the number of features

    c_w_ : array-like, shape = [n_prototypes]
        Prototype classes

    classes_ : array-like, shape = [n_classes]
        Array containing labels.

    dim_ : int
        Maximum rank or projection dimensions

    omega_ : array-like, shape = [dim, n_features]
        Relevance matrix

    See also
    --------
    GlvqModel, GrlvqModel, LgmlvqModel
    """

    # ptype_id = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    # prototypes_per_class = 2
    gaussian_sd = 0.5
    gaussian_sd_wrong = 1
    kernel_size = 1
    # omega_ = np.eye(2)
    # W_ = np.array([[1.1, 9.5], [2.1, 9.8], [20, 20], [32, -1.7], [0.5, -2.5], [9.8, -3.4], [-9.4, -3.6]
    #                             , [-2.8, -9.2], [-3.1, -9.5], [-3.7, 9.9], [-9.9, 0.1], [-0.5, 0.5]])

    def __init__(self, prototypes_per_class=1, kernel_size=1, initial_prototypes=None,
                 initial_matrix=None, regularization=0.0,
                 dim=None, max_iter=2500, gtol=1e-5, display=False,
                 random_state=None, lr_prototype=0.1, lr_omega=0.05):
        super(OGmlvqModel, self).__init__(prototypes_per_class,
                                         initial_prototypes, max_iter, gtol,
                                         display, random_state)
        self.regularization = regularization
        self.initial_matrix = initial_matrix
        self.initialdim = dim
        self.kernel_size = kernel_size
        self.gtol = gtol
        self.lr_prototype = lr_prototype
        self.lr_omega = lr_omega

    def find_prototype(self, data_point, label, k_size):
        list_square_dist = _squared_euclidean(data_point.dot(self.omega_.T), self.w_.dot(self.omega_.T)).flatten()
        # list_square_dist = _squared_euclidean(data_point.dot(self.omega_), self.w_.dot(self.omega_)).flatten()
        list_dist = np.sqrt(list_square_dist)
        np.around(list_dist, decimals=5)

        # list_square_dist2 = self._compute_distance(data_point, self.w_, self.omega_)
        # list_dist2 = np.sqrt(list_square_dist2)
        # print(list_dist2)

        correct_cls_min = label - k_size
        correct_cls_max = label + k_size
        if correct_cls_min < self.ranking_range[0]:
            correct_cls_min = self.ranking_range[0]
        if correct_cls_max > self.ranking_range[1]:
            correct_cls_max = self.ranking_range[1]

        correct_ranking = np.array(list(range(int(correct_cls_min), int(correct_cls_max) + 1)))

        # all classes with True and False
        class_list = np.zeros((len(self.c_w_)//self.prototypes_per_class), dtype=bool)
        class_list[correct_ranking] = True

        # print(list_dist, label, self.c_w_)
        # correct kernel class
        correct_idx0 = correct_cls_min * self.prototypes_per_class
        correct_idx1 = correct_cls_max * self.prototypes_per_class + self.prototypes_per_class
        proto_correct_list = np.array(list(range(int(correct_idx0), int(correct_idx1))))

        prototype_list = np.zeros((len(self.c_w_)), dtype=bool)
        prototype_list[proto_correct_list] = True

        D = list_dist[np.invert(prototype_list)].mean()
        # print(class_list)

        # collection set of closest prototype from each correct class
        # collection set of closest prototype from each wrong class
        cls_ind = 0
        W_plus = []
        W_minus = []
        max_error_cls = 0
        for correct_cls in class_list:
            ind0 = cls_ind * self.prototypes_per_class
            ind1 = ind0 + self.prototypes_per_class
            min_val = min(list_dist[ind0:ind1])
            min_idx = np.argmin(list_dist[ind0:ind1], axis=0) + ind0
            # print(min_idx, min_val)
            if correct_cls:
                W_plus.append([min_idx, min_val])
            elif min_val <= D:
                W_minus.append([min_idx, min_val])
                if abs(cls_ind - label) > max_error_cls:
                    max_error_cls = abs(cls_ind - label)
            cls_ind += 1
        # print(W_plus, W_minus)
        return W_plus, W_minus, max_error_cls, D

    # update prototype a and b, and omega
    def update_prot_and_omega(self, w_plus, w_minus, label, max_error_cls, datapoint, lr_pt, lr_om, D):
        # print(w_plus, w_minus)
        while len(w_plus) > 0 and len(w_minus) > 0:
            # find closest correct prototype from w_plus
            min_value = np.inf
            min_ind_correct = 0
            index = 0
            for prot in w_plus:
                if prot[1] < min_value:
                    min_value = prot[1]
                    min_ind_correct = index
                index += 1
            closest_cor_p = w_plus.pop(min_ind_correct)

            # find closest wrong prototype from w_minus
            min_value = np.inf
            min_ind_wrong = 0
            index = 0
            for prot in w_minus:
                if prot[1] < min_value:
                    min_value = prot[1]
                    min_ind_wrong = index
                index += 1
            closest_wro_p = w_minus.pop(min_ind_wrong)

            # update prototypes and omega here
            pt_pair = [closest_cor_p, closest_wro_p]
            delta_correct_prot, delta_wrong_prot, delta_omega = self._derivatives(pt_pair, label, max_error_cls, datapoint, D)

            pid_correct = closest_cor_p[0]
            pid_wrong = closest_wro_p[0]
            # print(self.w_)
            # print(self.omega_)
            self.w_[pid_correct] = self.w_[pid_correct] + delta_correct_prot * lr_pt
            self.w_[pid_wrong] = self.w_[pid_wrong] + delta_wrong_prot * lr_pt
            self.omega_ = self.omega_ - delta_omega * lr_om
            # print(self.w_)
            # print(self.omega_)

    # calculate derivatives of prototypes a, b and omega
    def _derivatives(self, pt_pair, label, max_error_cls, datapoint, D):
        # print(max_error_cls)
        # calculate alpha+ and alpha-
        alpha_distance_plus, alpha_plus = self.alpha_dist_plus(pt_pair, label)
        alpha_distance_minus, alpha_minus = self.alpha_dist_minus(pt_pair, label, max_error_cls, D)

        gamma_plus = 2*alpha_distance_minus / pow((alpha_distance_plus + alpha_distance_minus), 2)
        gamma_minus = -2*alpha_distance_plus / pow((alpha_distance_plus + alpha_distance_minus), 2)

        pid_correct = pt_pair[0][0]
        pid_wrong = pt_pair[1][0]
        diff_correct = datapoint - self.w_[pid_correct]
        diff_wrong = datapoint - self.w_[pid_wrong]

        squared_sum_alpha_plus_minus = (pow((alpha_distance_plus+alpha_distance_minus), 2))
        mu_plus = 2*alpha_distance_minus*alpha_plus/squared_sum_alpha_plus_minus
        mu_minus = 2*(1-pt_pair[1][1]/(2*pow(self.gaussian_sd_wrong, 2)))*alpha_minus*alpha_distance_plus/squared_sum_alpha_plus_minus

        diff_mtx_correct = diff_correct.T.dot(diff_correct)
        delta_omega_plus = gamma_plus * 2 * alpha_plus * self.omega_.dot(diff_mtx_correct)

        diff_mtx_wrong = diff_wrong.T.dot(diff_wrong)
        delta_omega_minus = gamma_minus * 2 * alpha_minus * self.omega_.dot(diff_mtx_wrong)

        # delta_omega = delta_omega_plus + delta_omega_minus
        # print("delta_omega:", delta_omega)
        delta_omega = 2 * mu_plus * self.omega_.dot(diff_mtx_correct) - 2*mu_minus*self.omega_.dot(diff_mtx_wrong)

        # delta_correct_prot = gamma_plus * (-2*alpha_plus*diff_correct.dot(self.omega_.T.dot(self.omega_)))
        delta_correct_prot = 2*mu_plus * diff_correct.dot(self.omega_.T.dot(self.omega_))

        # addition_term = -2*alpha_minus*(-1/(2*pow(self.gaussian_sd_wrong, 2))) * pt_pair[1][1] * diff_wrong.dot(self.omega_.T.dot(self.omega_))
        # delta_wrong_prot = gamma_minus * (-2*alpha_minus*diff_wrong.dot(self.omega_.T.dot(self.omega_))+addition_term)
        # print("delta:", delta_correct_prot, delta_wrong_prot)
        delta_wrong_prot = -2*mu_minus*diff_wrong.dot(self.omega_.T.dot(self.omega_))

        return delta_correct_prot, delta_wrong_prot, delta_omega

    def alpha_dist_plus(self, pt_pair, label):
        distance_correct = pt_pair[0][1]
        ranking_diff_correct = abs(label - pt_pair[0][0] // self.prototypes_per_class)

        alpha_plus = math.exp(- pow(ranking_diff_correct, 2) / (2 * pow(self.gaussian_sd, 2)))

        alpha_distance_plus = alpha_plus * distance_correct

        return alpha_distance_plus, alpha_plus

    def alpha_dist_minus(self, pt_pair, label, max_error_cls, D):
        distance_wrong = pt_pair[1][1]
        ranking_diff_wrong = abs(label - pt_pair[1][0] // self.prototypes_per_class)

        alpha_minus = math.exp(- pow(max_error_cls - ranking_diff_wrong, 2) / (2 * pow(self.gaussian_sd, 2))) \
                      * \
                      math.exp(-pow(distance_wrong, 2) / (2 * pow(self.gaussian_sd_wrong, 2)))

        alpha_distance_minus = alpha_minus * distance_wrong

        return alpha_distance_minus, alpha_minus

    def _costfunc(self, data_point, label, k_size):
        w_plus, w_minus, max_error_cls, D = self.find_prototype(data_point, label, k_size)

        sum_cost = 0
        cost_count = 0
        while len(w_plus) > 0 and len(w_minus) > 0:
            min_value = np.inf
            min_ind_correct = 0
            index = 0
            for prot in w_plus:
                if prot[1] < min_value:
                    min_value = prot[1]
                    min_ind_correct = index
                index += 1
            closest_cor_p = w_plus.pop(min_ind_correct)

            # find closest wrong prototype from w_minus
            min_value = np.inf
            min_ind_wrong = 0
            index = 0
            for prot in w_minus:
                if prot[1] < min_value:
                    min_value = prot[1]
                    min_ind_wrong = index
                index += 1
            closest_wro_p = w_minus.pop(min_ind_wrong)

            # update prototypes and omega here
            pt_pair = [closest_cor_p, closest_wro_p]
            alpha_distance_plus, alpha_plus = self.alpha_dist_plus(pt_pair, label)
            alpha_distance_minus, alpha_minus = self.alpha_dist_minus(pt_pair, label, max_error_cls, D)
            mu = (alpha_distance_plus - alpha_distance_minus) / (alpha_distance_plus + alpha_distance_minus)
            sum_cost += mu
            cost_count += 1

        return sum_cost, cost_count

    def _optimize(self, x, y, random_state):
        if not isinstance(self.regularization, float) or self.regularization < 0:
            raise ValueError("regularization must be a positive float ")
        nb_prototypes, nb_features = self.w_.shape
        if self.initialdim is None:
            self.dim_ = nb_features
        elif not isinstance(self.initialdim, int) or self.initialdim <= 0:
            raise ValueError("dim must be an positive int")
        else:
            self.dim_ = self.initialdim

        self.ranking_range = [int(self.classes_.min()), int(self.classes_.max())]
        self.ranking_list = np.array(list(range(self.ranking_range[0], self.ranking_range[1] + 1)))

        if self.initial_matrix is None:
            if self.dim_ == nb_features:
                self.omega_ = np.eye(nb_features)
            else:
                self.omega_ = random_state.rand(self.dim_, nb_features) * 2 - 1
        else:
            self.omega_ = validation.check_array(self.initial_matrix)
            if self.omega_.shape[1] != nb_features:  # TODO: check dim
                raise ValueError(
                    "initial matrix has wrong number of features\n"
                    "found=%d\n"
                    "expected=%d" % (self.omega_.shape[1], nb_features))

        self.gaussian_sd = self.gaussian_sd * math.sqrt(nb_features)


        # start the algorithm
        stop_flag = False
        epoch_index = 0
        max_epoch = 100
        cost_list = np.zeros([max_epoch, 1])
        lr_pt = self.lr_prototype
        lr_om = self.lr_omega
        while not stop_flag:

            for i in range(len(x)):
                index = random.randrange(len(x))
                datapoint = np.array([x[index]])
                label = y[index]
                W_plus, W_minus, max_error_cls, D = self.find_prototype(datapoint, label, self.kernel_size)
                self.update_prot_and_omega(W_plus, W_minus, label, max_error_cls, datapoint, lr_pt, lr_om, D)
                # normalize the omega
                self.omega_ /= math.sqrt(
                    np.sum(np.diag(self.omega_.T.dot(self.omega_))))

            sum_cost = 0
            cost_count = 0
            for index in range(len(x)):
                datapoint = np.array([x[index]])
                label = y[index]
                cost, count = self._costfunc(datapoint, label, self.kernel_size)
                sum_cost += cost
                cost_count += count

            cost_list[epoch_index] = sum_cost/cost_count
            epoch_index += 1
            if epoch_index >= max_epoch:
                stop_flag = True
                print(cost_list)

            lr_pt = lr_pt / (1 + self.gtol * (epoch_index - 1))
            lr_om = lr_om / (1 + self.gtol * (epoch_index - 1))

        # variables = np.append(self.w_, self.omega_, axis=0)
        # label_equals_prototype = y[np.newaxis].T == self.c_w_
        # method = 'l-bfgs-b'
        # res = minimize(
        #     fun=lambda vs:
        #     self._optfun(vs, x, label_equals_prototype=label_equals_prototype),
        #     jac=lambda vs:
        #     self._optgrad(vs, x, label_equals_prototype=label_equals_prototype,
        #                   random_state=random_state,
        #                   lr_prototypes=1, lr_relevances=0),
        #     method=method, x0=variables,
        #     options={'disp': self.display, 'gtol': self.gtol,
        #              'maxiter': self.max_iter})
        # n_iter = res.nit
        #
        # out = res.x.reshape(res.x.size // nb_features, nb_features)
        # self.w_ = out[:nb_prototypes]
        # self.omega_ = out[nb_prototypes:]
        # self.omega_ /= math.sqrt(
        #     np.sum(np.diag(self.omega_.T.dot(self.omega_))))
        # self.n_iter_ = n_iter

    def _compute_distance(self, x, w=None, omega=None):
        if w is None:
            w = self.w_
        if omega is None:
            omega = self.omega_
        nb_samples = x.shape[0]
        nb_prototypes = w.shape[0]
        distance = np.zeros([nb_prototypes, nb_samples])
        for i in range(nb_prototypes):
            distance[i] = np.sum((x - w[i]).dot(omega.T) ** 2, 1)
        return distance.T

    def score(self, x, y):
        count = 0
        for i in range(len(x)):
            datapoint = np.array([x[i]])
            distance_list = _squared_euclidean(datapoint.dot(self.omega_.T), self.w_.dot(self.omega_.T)).flatten()
            min_ind = np.argmin(distance_list, axis=0)
            predict_class = self.c_w_[min_ind]

            if abs(predict_class - y[i]) <= self.kernel_size:
                count += 1

        accuracy = count/len(x)
        # print("accuracy", accuracy)
        return accuracy

    def project(self, x, dims, print_variance_covered=False):
        """Projects the data input data X using the relevance matrix of trained
        model to dimension dim

        Parameters
        ----------
        x : array-like, shape = [n,n_features]
          input data for project
        dims : int
          dimension to project to
        print_variance_covered : boolean
          flag to print the covered variance of the projection

        Returns
        --------
        C : array, shape = [n,n_features]
            Returns predicted values.
        """
        v, u = np.linalg.eig(self.omega_.conj().T.dot(self.omega_))
        idx = v.argsort()[::-1]
        if print_variance_covered:
            print('variance coverd by projection:',
                  v[idx][:dims].sum() / v.sum() * 100)
        return x.dot(u[:, idx][:, :dims].dot(np.diag(np.sqrt(v[idx][:dims]))))
