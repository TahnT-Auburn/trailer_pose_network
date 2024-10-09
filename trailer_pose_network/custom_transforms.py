'''
################### Custom Transforms ###################

A script that houses custom transforms for setting up 
PyTorch Data.
TODO: Move to a separate Network Utils package

Author: Tahn Thawainin, AU GAVLAB
        email: pzt0029@auburn.edu
        github: https://github.com/TahnT-Auburn

#########################################################
'''
#%%
import torch
import cv2
import numpy as np

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Arguments:
        image: Input image to rescale.
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_dims = (int(new_h), int(new_w))

        new_image = cv2.resize(image, new_dims)

        return new_image    