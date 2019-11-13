# Python Flask Stylegan Presenter
#
# MCW Colab Research
#

#
# imports
#
import os
import io
from flask import Flask, render_template, request, Response, send_file
import pickle
import scipy
import numpy as np
import tensorflow as tf
import PIL.Image
import dnnlib # local nvidia module
from dnnlib import tflib as tflib

# 
# config params
#
pkl_path = os.path.join(os.getcwd(),'pkls/network-final.pkl')
output_dir = os.path.join(os.getcwd(),'static/generated_images')
test_img_src = os.path.join(os.getcwd(),'static/generated_images/test_fake_img.png')
seed = 422

#
# globals
#

app = Flask(__name__)
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

#
# neural network utility methods
#
def load_stylegan():
    tflib.init_tf()
    with open(pkl_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)

stylegan_network = None
def get_stylegan_network():
    global stylegan_network
    if stylegan_network == None:
        stylegan_network = load_stylegan()
    return stylegan_network

def get_gs():
    return get_stylegan_network()[2]

#
# methods

def _handle_stylegan_generate_image_click(slider_values, blob_image, generateNewImage=True):
    '''
    @params generateNewImage if true, try to use tensorflow generator. if false, use static route (test_img_src) on api call.
    '''
    if generateNewImage:
        generate_random_png(slider_values, blob_image, random=True)


def generate_random_png(slider_values, blob_image, seed=seed, random=True):
        if random:
            input_seed = int(slider_values[0]*1000)
            seed = (np.random.RandomState(input_seed).rand(1)*1000).astype('int')
        Gs = get_gs()
        _make_png(blob_image, Gs, slider_values, seed=seed)

def _make_png(image_object, Gs, slider_values, cx=0, cy=0, cw=1024, ch=1024, seed = seed):
    print('attempting to create png')
    latents = make_latents(Gs, seed, slider_values)
    images = Gs.run(latents, None, **synthesis_kwargs) # [seed, y, x, rgb]
    print(f'length of generated images vector: {len(images)}')
    image = PIL.Image.fromarray(images[0], 'RGB') # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray
    image.save(image_object, format="PNG")
    print('success')

def make_latents(Gs, seed, slider_values):
  sigma_scaler = slider_values[2]
  sigma = slider_values[1]*1000
  mu_scaler = slider_values[4]
  mu = slider_values[3]*1000
  print(f'sigma: {sigma}')
  print(f'sigma scale: {sigma_scaler}')
  print(f'mu: {mu}')
  print(f'mu_scaler: {mu_scaler}')
  latents =  sigma_scaler * sigma * np.random.RandomState(seed).randn(1, Gs.input_shape[1]) + mu * mu_scaler
#   latent_index = int((slider_values[4])*1000)
#   latents =  sigma * get_latent_vector_from_all_latents(Gs, seed, latent_index) + mu
  return latents

# all_latents = 'None'
# def get_latent_vector_from_all_latents(Gs, seed, latent_index):
#     global all_latents
#     print('latent index: ', latent_index)
#     if all_latents == 'None':
#       print('Generating latent vectors...')
#       shape = [1000, np.prod([1,1])] + Gs.input_shape[1:] # [num_latent_codes, image?, color_channel, component?]
#       all_latents = np.random.RandomState(seed).randn(*shape).astype(np.float32)
#       # sigma = [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape)
#     #   sigma = [1000] + [0] * len(Gs.input_shape)
#       sigma = 10
#       all_latents = scipy.ndimage.gaussian_filter(all_latents, sigma, mode='wrap')
#       all_latents /= np.sqrt(np.mean(np.square(all_latents)))
#     return all_latents[latent_index]

def trim_out_full_path(full_path):
    '''
    internal wiring method
    python flask server can only access urls from static dir
    '''
    match_string = '/generated_images/'
    index = full_path.find(match_string)
    return full_path[index:]

def printDebug(phrase):
    '''
    simple method to print debug output
    '''
    print(f'DEBUG | {phrase}')

#load_stylegan
# routes
#
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def api():
    slider_values = [float(val) for val in request.get_json()]
    print(slider_values)
    # host = 'http://cc-build01.hpc.msoe.edu'
    host = 'http://localhost:5000/static'
    return f'{host}{_handle_stylegan_generate_image_click(slider_values, generateNewImage=True)}'
    # return 'https://www.odt.co.nz/sites/default/files/styles/odt_story_slideshow/public/slideshow/node-374554/2016/04/two_mako_sharks_fighting__56d00b9c7e.JPG?itok=ShwKtfMp'

#
# main
#
if __name__ == "__main__":
    print('>> Starting flask_presenter.py main method')
    app.run(debug=True, host='0.0.0.0', threaded=False)