import matplotlib.pyplot as plt
import tensorflow as tf
import os
import glob
import pickle
try:
    from util_images import plot_image_with_mask
    from model_new import segmenter
except:
    from src.util_images import plot_image_with_mask
    from src.model_new import segmenter
import numpy as np
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../output/')
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../models/')

def training_curves (results) :
    """ Displays accuracy on training and validation batches after each epoch"""

    plt.figure()
    for key in results.history.keys() :
        plt.plot(results.history[key])
        plt.plot(results.history[key], label=key)
        plt.title('Training curves')
        plt.legend()

    plt.show()
    return

def load_training_sessions():
    for res_path in glob.glob(os.path.join(model_dir,'*.pck')):
        res = pickle.load(open(res_path, 'rb'))
        print(f'{res.id_model} {res.img_dims} {res.lambd} {res.architecture}')
    return


def predict_example_and_plot(model, X, Y, size):
    for i in range(len(X)):
        Y_pred = model.predict(X[i].reshape((1,size, size, 1))) > 0.5
        plot_image_with_mask(X[i], Y[i], pred_mask=Y_pred, size = size)
    return

def plot_param_search(dir = 'archit', names = {18069: 'dropout = 0.2'}, title = None):
    if title is None:
        title = dir
    fig = plt.figure(figsize=(10,5))
    for id in names.keys():
        res_path = os.path.join(model_dir,f'{dir}/{id}.pck')
        res = pickle.load(open(res_path, 'rb'))
        plt.plot(res['training_results']['loss'], label = f'{names[id]} final score: {res.score[1]}')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(title)
    plt.savefig(os.path.join(output_dir,f'{title}.png'))
    return


if __name__ == '__main__':
    load_training_sessions()