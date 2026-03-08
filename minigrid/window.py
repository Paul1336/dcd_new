# Copyright (c) 2019 Maxime Chevalier-Boisvert.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    print('To display the environment in a window, please install matplotlib, eg:')
    print('pip3 install --user matplotlib')
    sys.exit(-1)

class Window:
    """Window to draw a gridworld instance using Matplotlib."""

    def __init__(self, title):
        self.fig = None
        self.imshow_obj = None
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(title)
        plt.axis('off')
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def show_img(self, img):
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img, interpolation='bilinear')
        self.imshow_obj.set_data(img)
        self.fig.canvas.draw()
        plt.pause(0.001)

    def set_caption(self, text):
        plt.xlabel(text)

    def reg_key_handler(self, key_handler):
        self.fig.canvas.mpl_connect('key_press_event', key_handler)

    def show(self, block=True):
        if not block:
            plt.ion()
        plt.show()

    def close(self):
        plt.close()
