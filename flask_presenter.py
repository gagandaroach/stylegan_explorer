# Python Flask Stylegan Presenter
#
# Milwaukee School of Engineering and Medical College of Wisconsin
# Collaborative Research
# Gagan Daroach <gagandaroach@gmail.com>

#
# imports
#
import os
import pickle
import scipy
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

# Note: This import depends on how PIL is installed on your system.
# import PIL.Image
import Image

# Import Nvidia StyleGAN Network Libraries
import dnnlib
from dnnlib import tflib as tflib

#
# config params
#
cwd = os.getcwd()
pkl_path = os.path.join(cwd, 'pkls/network-final.pkl')
output_dir = os.path.join(cwd, 'static/generated_images')
test_img_src = os.path.join(cwd, 'static/generated_images/test_fake_img.png')
seed = 422

#
# globals
#
app = Flask(__name__)
synthesis_kwargs = dict(output_transform=dict(
    func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
stylegan_network = None

#
# neural network utility methods
#
def load_stylegan():
    tflib.init_tf()
    with open(pkl_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)


def get_stylegan_network():
    global stylegan_network
    if stylegan_network == None:
        stylegan_network = load_stylegan()
    return stylegan_network


def get_gs():
    return get_stylegan_network()[2]

#
# StyleGAN Explorer Core Methods
#
def _handle_stylegan_generate_image_click(slider_values, generateNewImage=True):
    '''
    @params generateNewImage if true, try to use tensorflow generator. if false, use static route (test_img_src) on api call.
    '''
    if generateNewImage:
        full_path = generate_random_png(slider_values, random=True)
    else:
        full_path = test_img_src

    trimmed_path = trim_out_full_path(full_path)
    printDebug('_handle_stylegan_generate_image_click returning: %s' %
               trimmed_path)
    return trimmed_path  # todo, add full path, generate time in json obj here for return


def generate_random_png(slider_values, seed=seed, random=True):
    if random:
        input_seed = int(slider_values[0]*1000)
        seed = (np.random.RandomState(input_seed).rand(1)*1000).astype('int')
    # seed, sigma, mu
    filename = f'output_{str(seed)}_{slider_values[1]}_{slider_values[2]}_{slider_values[3]}_{slider_values[4]}.png'
    Gs = get_gs()
    return _make_png(os.path.join(output_dir, filename), Gs, slider_values, seed=seed)


def _make_png(png_path, Gs, slider_values, cx=0, cy=0, cw=1024, ch=1024, seed=seed):
    printDebug('attempting to create png at %s' % png_path)
    latents = make_latents(Gs, seed, slider_values)
    images = Gs.run(latents, None, **synthesis_kwargs)
    printDebug(f'Length of generated images vector: {len(images)}')
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray
    image = Image.fromarray(images[0], 'RGB')
    image.save(png_path)
    printDebug(f'Successfully saved @ {png_path}')
    return png_path


def make_latents(Gs, seed, slider_values):
    '''
    Generat Latent Vectors
    In Slider Values
      0: Seed
      1. Sigma 
      2. Sigma Scalar
      3. Mu
      4. Mu Scalar
    '''
    sigma_scaler = slider_values[2]
    sigma = slider_values[1]*1000
    mu_scaler = slider_values[4]
    mu = slider_values[3]*1000
    printDebug(f'sigma: {sigma}')
    printDebug(f'sigma_scaler: {sigma_scaler}')
    printDebug(f'mu: {mu}')
    printDebug(f'mu_scaler: {mu_scaler}')

    sigma = sigma_scaler * sigma
    mu = mu * mu_scaler
    latent_vector_length = Gs.input_shape[1]
    latents = generate_normal_distribution(sigma, mu, latent_vector_length)
    return latents


def generate_normal_distribution(sigma, mu, length):
    '''
    Generate Normal Distribution of Points
    https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randn.html
    '''
    return sigma * np.random.randn(1, length) + mu


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

#
# flask routes
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
