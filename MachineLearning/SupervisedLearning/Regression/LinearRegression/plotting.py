# Jack C. Cook
# Wednesday, September 2, 2020

# a module containing any relevant plotting functions

import matplotlib.pyplot as plt
import numpy as np


class PyPlotter:
    def __init__(self, row=1, col=1):
        fig, ax = plt.subplots(row, col)
        self.fig = fig
        self.ax = ax

    def plot(self, x: np.ndarray, y: np.ndarray, label='', close=True, show=True, save_fig=True,
             fig_name='output', fig_ext='jpg', marker='o', plot_style='scatter', line_style='-') -> None:
        ax = self.ax
        fig = self.fig
        if plot_style == 'scatter':
            if label == '':
                ax.scatter(x, y)
            else:
                ax.scatter(x, y, label=label, marker=marker)
        elif plot_style == 'plot':
            if label == '':
                ax.plot(x, y)
            else:
                ax.plot(x, y, label=label, marker=marker, ls=line_style)

        # get the legend entries and update the legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best')

        self.add_labels()  # adding labels, trapped in function # TODO: make this accessible

        if save_fig is True:
            self.save_figure(fig_name=fig_name, fig_ext=fig_ext)

        if show is True:
            plt.show()

        if close is True:
            plt.close(fig)

        return

    def save_figure(self, fig_name='output', fig_ext='jpg'):
        self.fig.savefig(fig_name + '.' + fig_ext)

    def add_labels(self, x='x', y='y', title='') -> None:
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)
        if title != '':
            self.ax.set_title(title)
        return


# class PlotAttributes:
#     def __init__(self, x='x', y='y', label=None, title=None):
#         self.x = x
#         self.y = y
#         self.label = label
#         self.title = title
#
#     def modify_plot(self, ax):
#         ax.set_xlabel(self.x)
#         ax.set_ylabel(self.y)
#         if self.title is not None:
#             ax.set_title(self.title)
#
#
# def plot(x: np.ndarray, y: np.ndarray, plat=None, close=True, show=True, save_fig=True, fig_name='output', fig_ext='jpg'):
#     fig, ax = plt.subplots()
#     ax.scatter(x, y)
#     if plat is not None:
#         plat.modify_plot(ax)
#
#     if save_fig is True:
#         fig.savefig(fig_name + '.' + fig_ext)
#
#     if show is True:
#         plt.show()
#
#     if close is True:
#         plt.close(fig)
#     else:
#         return fig, ax
#
#     return
