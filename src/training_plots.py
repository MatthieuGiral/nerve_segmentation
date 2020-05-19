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
    for res_path in glob.glob(os.path.join(model_dir,'lambdas/*.pck')):
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
        res_path = os.path.join(model_dir,f'{dir}/model_{id}.pck')
        res = pickle.load(open(res_path, 'rb'))
        plt.plot(res.training_results['loss'], label = f'{names[id]} final test score: {res.score[1]:.2f}')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(output_dir,f'{title}.png'))
    return


if __name__ == '__main__':
    load_training_sessions()
    plot_param_search('dropout',
                      {32695:'dropout 0.4',
                       57459:'dropout 0.2',
                       61022:'dropout 0.1',
                       66569:'no dropout'},
                      'dropout_graph')
    plot_param_search('architecture',
                      {55403: 'architecture [512,256,128,64]',
                       66569: 'architecture [1024,512,256,128,64]',
                       93655: 'architecture [1024,512,256,128,64,32]',
                       59849: 'architecture [256,128,64]'},
                      'architecture_graph')
    plot_param_search('augmentation',
                      {66569: 'No augmentation',
                       75870: 'flip & mirror'},
                      'augmentation')
    # plot_param_search('lambdas',
    #                   {68876: 'lam = 1e-5',
    #                    74659: 'lam = 1e-2',
    #                    99523: 'lam = 1e-3',
    #                    99990: 'lam = 1e-4'},
    #                   'lambdas')

