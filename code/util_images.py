import tensorflow as tf
import glob
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

def plot_image(img, title=None):
    plt.figure(figsize=(15, 20))
    plt.title(title)
    plt.imshow(img)
    plt.show()

def fimg_to_fmask(img_path):
    """
    >>> fimg_to_fmask('../data/train/36_62.tif')
    '../data/train/19_81_mask.tif'

    @param img_path: the path of the image you want to get the mask from
    @return:path of the mask
    """
    # convert an image file path into a corresponding mask file path
    dirname, basename = os.path.split(img_path)
    maskname = basename.replace(".tif", "_mask.tif")
    return os.path.join(dirname, maskname)

def image_with_mask(img, mask):
    # returns a copy of the image with edges of the mask added in red
    img_color = np.dstack((img, img, img))
    mask_edges = cv2.Canny(mask, 100, 200) > 0
    img_color[mask_edges, 0] = 255  # set channel 0 to bright red, green & blue channels to 0
    img_color[mask_edges, 1] = 0
    img_color[mask_edges, 2] = 0
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
    plot_image(image_with_mask(img, mask), title=f_combined)
    print('plotted:', f_combined)
    return

def get_annotated_data(n_images,
                       show_images = False):
    """
    Read n_images and transform it into arrays
    @param n_images: number of images to be fetched
    @param show_images: whether or not to show the images which are loaded
    @return: (X, Y): Arrays of shape (img.shape[0], img.shape[0], n_images) \
                            which represents the images and the associated masks
    """
    f_ultrasounds = [img for img in glob.glob("../data/train/*.tif") if 'mask' not in img][:n_images]
    f_masks = [fimg_to_fmask(fimg) for fimg in f_ultrasounds][:n_images]
    if show_images:
        for f_ultrasound in f_ultrasounds:
            show_image_with_mask(f_ultrasound)
    imgs = [np.asarray(Image.open(f_ultrasound)) for f_ultrasound in f_ultrasounds]
    masks = [np.asarray(Image.open(f_mask)) for f_mask in f_masks]
    X = np.dstack(imgs)
    Y = np.dstack(masks)
    return X, Y

if __name__ == '__main__':
    X, Y = get_annotated_data(20)
    print('ok')