import tensorflow as tf
import glob
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../data')

def plot_image(img, title=None):
    plt.figure(figsize=(15, 20))
    plt.title(title)
    plt.imshow(img)
    plt.show()

def fimg_to_fmask(img_path):
    """

    @param img_path: the path of the image you want to get the mask from
    @return:path of the mask
    """
    # convert an image file path into a corresponding mask file path
    dirname, basename = os.path.split(img_path)
    maskname = basename.replace(".tif", "_mask.tif")
    return os.path.join(dirname, maskname)

def plot_image_with_mask(img, mask, **kwargs):
    # returns a copy of the image with edges of the mask added in red
    img_color = np.dstack((img, img, img))
    mask_edges = cv2.Canny(np.array(mask), 100, 200) > 0
    for k in kwargs.keys():
        key_mask_edges = cv2.Canny(np.array(np.array(kwargs[k]), 100, 200)) > 0
        color_k = list(np.random.choice(range(256), size=3))
        img_color[key_mask_edges,:] = color_k
    img_color[mask_edges, 0] = 255  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
    plt.imshow(img_color)
    return img_color

def show_image_with_mask(img_path):
    """
    Show the image in the path given with its mask
    @param img_path: location of the image e.g.: '../data/train/19_81_mask.tif'
    @return: None
    """
    mask_path = fimg_to_fmask(img_path)
    img = plt.imread(img_path)
    mask = plt.imread(mask_path)
    f_combined = img_path + " & " + mask_path
    plot_image_with_mask(img, mask)
    print('plotted:', f_combined)
    return

def get_annotated_data(n_images,
                       show_images = False,
                       new_size = None):
    """
    Read n_images and transform it into arrays

    >>> get_annotated_data(10, \
                            new_size = (520,520), \
                            show_images = True)[0].shape == (10, 520, 520, 1)
    True


    @param n_images: number of images to be fetched
    @param show_images: whether or not to show the images which are loaded
    @param new_size: if you want the image to be resized to a specific size, specify a tuple (img_height, img_width)
    @return: (X, Y): Arrays of shape (n_images, img.shape[0], img.shape[1], 1) \
                            which represents the images and the associated masks
    """
    f_ultrasounds = [img for img in glob.glob(os.path.join(data_dir,"train/*.tif")) if 'mask' not in img][:n_images]
    f_masks = [fimg_to_fmask(fimg) for fimg in f_ultrasounds][:n_images]
    imgs = [Image.open(f_ultrasound) for f_ultrasound in f_ultrasounds]
    masks = [Image.open(f_mask) for f_mask in f_masks]

    if new_size is not None:
        imgs = [img.resize(new_size) for img in imgs]
        masks = [mask.resize(new_size) for mask in masks]
    else:
        new_size = imgs[0].size
    if show_images is True:
        for i in range(n_images):
            plot_image_with_mask(imgs[i], masks[i])
            plt.show()
    X = np.stack(imgs).reshape((n_images, new_size[0], new_size[1], 1)) / 255
    Y = np.stack(masks).reshape((n_images, new_size[0], new_size[1], 1)) / 255
    return X, Y

if __name__ == '__main__':
    import doctest
    doctest.testmod()