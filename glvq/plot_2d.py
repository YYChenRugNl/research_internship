from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import validation


def simple_line_plot(x, y, title, nb1, nb2, nb3, measure, ylabel, xlabel='Training epochs', set_y=False):
    subplot_nb = nb1*100+nb2*10+nb3
    f = plt.figure(1, figsize=(7, 7))
    # f.suptitle(title)
    ax = f.add_subplot(subplot_nb)
    if set_y and measure == 'MZE':
        ax.set_ylim([0.4, 1])
    elif set_y and measure == 'MAE':
        ax.set_ylim([0.8, 2])
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    f.show()
    # plt.plot(x, y)
    # plt.show()


def plot2d(model, x, y, proto_history_list=[], figure=1, title="", prototype_count=-1, no_index=False):
    """
    Projects the input data to two dimensions and plots it. The projection is
    done using the relevances of the given glvq model.

    :param model: GlvqModel that has relevances
        (GrlvqModel,GmlvqModel,LgmlvqModel)
    :param x: Input data
    :param y: Input data target
    :param figure: the figure to plot on
    :param title: the title to use, optional
    :return: None
    """
    x, y = validation.check_X_y(x, y)
    model.init_w, model.c_w_ = validation.check_X_y(model.init_w, model.c_w_)
    model.w_, model.c_w_ = validation.check_X_y(model.w_, model.c_w_)
    dim = 2
    f = plt.figure(figure, figsize=(10, 10))
    f.suptitle(title)
    pred = model.predict(x)

    if hasattr(model, 'omegas_'):
        nb_prototype = model.w_.shape[0]
        if prototype_count is -1:
            prototype_count = nb_prototype
        if prototype_count > nb_prototype:
            print(
                'prototype_count may not be bigger than number of prototypes')
            return
        ax = f.add_subplot(1, nb_prototype + 1, 1)
        ax.scatter(x[:, 0], x[:, 1], c=to_tango_colors(y, no_index=no_index), alpha=0.5)
        ax.scatter(x[:, 0], x[:, 1], c=to_tango_colors(pred, no_index=no_index), marker='.')

        ax.scatter(model.w_[:, 0], model.w_[:, 1], c=tango_color('aluminium', 5), marker='D')
        ax.scatter(model.w_[:, 0], model.w_[:, 1], c=to_tango_colors(model.c_w_, 0, no_index=no_index), marker='.')

        ax.axis('equal')

        d = sorted([(model._compute_distance(x[y == model.c_w_[i]],
                                             model.w_[i]).sum(), i) for i in
                    range(nb_prototype)], key=itemgetter(0))
        idxs = list(map(itemgetter(1), d))
        for i in idxs:
            x_p = model.project(x, i, dim, print_variance_covered=True)
            w_p = model.project(model.w_[i], i, dim)
            ax = f.add_subplot(1, nb_prototype + 1, idxs.index(i) + 2)
            ax.scatter(x_p[:, 0], x_p[:, 1], c=to_tango_colors(y, 0, no_index=no_index),
                       alpha=0.2)
            # ax.scatter(X_p[:, 0], X_p[:, 1], c=pred, marker='.')
            ax.scatter(w_p[0], w_p[1],
                       c=tango_color('aluminium', 5), marker='D')
            ax.scatter(w_p[0], w_p[1],
                       c=tango_color(i, 0), marker='.')
            ax.axis('equal')

    else:
        ax = f.add_subplot(221)
        ax.scatter(x[:, 0], x[:, 1], c=to_tango_colors(y, no_index=no_index), alpha=0.5)
        ax.scatter(x[:, 0], x[:, 1], c=to_tango_colors(pred, no_index=no_index), marker='.')

        ax.scatter(model.w_[:, 0], model.w_[:, 1],
                   c=tango_color('aluminium', 5), marker='D')
        ax.scatter(model.w_[:, 0], model.w_[:, 1],
                   c=to_tango_colors(model.c_w_, 0, no_index=no_index), marker='.')
        ax.axis('equal')
        x_p = model.project(x, dim, print_variance_covered=True)
        w_p = model.project(model.w_, dim)

        # plot initial prototypes
        ax.scatter(model.init_w[:, 0], model.init_w[:, 1], c=tango_color('aluminium', 5), marker='D')
        ax.scatter(model.init_w[:, 0], model.init_w[:, 1], c=to_tango_colors(model.c_w_, 0, no_index=no_index),
                   marker='x')

        ax = f.add_subplot(222)
        ax.scatter(x_p[:, 0], x_p[:, 1], c=to_tango_colors(y, 0, no_index=no_index), alpha=0.5)
        # ax.scatter(X_p[:, 0], X_p[:, 1], c=pred, marker='.')
        ax.scatter(w_p[:, 0], w_p[:, 1],
                   c=tango_color('aluminium', 5), marker='D')
        ax.scatter(w_p[:, 0], w_p[:, 1], s=60,
                   c=to_tango_colors(model.c_w_, 0, no_index=no_index), marker='.')
        ax.axis('equal')

        # trace prototypes

        # f.plot([2,2], [3,3])
        # ax = f.add_subplot(133)
        # ax.plot([1, 2, 3, 4], [1, 4, 9, 16])
        # ax.axis([0, 6, 0, 20])
        if proto_history_list:
            ax = f.add_subplot(223)

            # plot initial prototypes
            ax.scatter(model.init_w[:, 0], model.init_w[:, 1], c=tango_color('aluminium', 5), marker='D')
            ax.scatter(model.init_w[:, 0], model.init_w[:, 1], c=to_tango_colors(model.c_w_, 0, no_index=no_index),
                       marker='x')

            proto_history_list = np.array(proto_history_list)
            fake_y = np.ones([len(proto_history_list[:, 0]), 1])

            for i in range(len(proto_history_list[0, :])):
                w_, label = validation.check_X_y(proto_history_list[:, i, :], fake_y)
                all_x = np.append(model.init_w[i, 0], w_[:, 0])
                all_y = np.append(model.init_w[i, 1], w_[:, 1])
                color = to_tango_colors(np.array([i//prototype_count]), 0, no_index=no_index)[0]
                ax.plot(all_x, all_y, c=color)
                # plt.plot()

    f.show()


colors = {
    "skyblue": ['#729fcf', '#3465a4', '#204a87'],
    "scarletred": ['#ef2929', '#cc0000', '#a40000'],
    "orange": ['#fcaf3e', '#f57900', '#ce5c00'],
    "plum": ['#ad7fa8', '#75507b', '#5c3566'],
    "chameleon": ['#8ae234', '#73d216', '#4e9a06'],
    "butter": ['#fce94f', 'edd400', '#c4a000'],
    "chocolate": ['#e9b96e', '#c17d11', '#8f5902'],
    "aluminium": ['#eeeeec', '#d3d7cf', '#babdb6', '#888a85', '#555753',
                  '#2e3436']
}

color_names = list(colors.keys())


def tango_color(name, brightness=0):
    if type(name) is int:
        if name >= len(color_names):
            name = name % len(color_names)
        name = color_names[name]
    if name in colors:
        return colors[name][brightness]
    else:
        raise ValueError('{} is not a valid color'.format(name))


def to_tango_colors(elems, brightness=0, no_index=False):
    if no_index:
        return [tango_color(int(e), brightness) for e in elems]
    else:
        elem_set = list(set(elems))
        return [tango_color(elem_set.index(e), brightness) for e in elems]
