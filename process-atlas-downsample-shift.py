import nilearn as ni
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import os
import sys
import numpy as np
import pandas as pd

from PIL import Image
from PIL import ImageOps
from PIL import ImagePalette

start = int(sys.argv[1])
end = int(sys.argv[2])

NLABELS = 18

data_dir = '/scratch/groups/jyeatman/samjohns-projects/data'
input_parc_dir = data_dir + '/parc-images-jitter'
input_curv_dir = data_dir + '/curv-images-jitter'
output_image_base_dir = data_dir + '/parc-images-jitter-processed'
output_curv_dir = output_image_base_dir + '/curv'
output_parc_dir = output_image_base_dir + '/parc'

os.makedirs(output_image_base_dir, exist_ok=True)
os.makedirs(output_curv_dir, exist_ok=True)
os.makedirs(output_parc_dir, exist_ok=True)

parc_fps = os.listdir(input_parc_dir)
curv_fps = os.listdir(input_curv_dir)

tab20 = cm.get_cmap('tab20')
tab20_colors = 255 * np.array(tab20.colors)
pal = np.concatenate(([[255., 255., 255.]], tab20_colors))


### Helper functions for shifts (translations) ###

def get_random_shift(px, fraction=0.1):
    shift_max = int(fraction * px)
    return np.random.randint(low=-shift_max, high=shift_max, size=(2,))


def shift(xin, yin, fraction=0.1, shifts=None, channel=False, returnparams=True):
    """ Generate random shifts (translations / affine transformations)
        in np arrays that represent images.
        Assumes images are square. Can have optional channel in dim 0.
    """
    assert fraction < 1.0
    px = xin.shape[1]

    if shifts is None:
        shifts = get_random_shift(px, fraction=fraction)

    s1, s2 = shifts
    left = s1 < 0
    up = s2 < 0
    right = not left
    down = not up

    new_x = np.zeros_like(xin)
    new_y = np.zeros_like(yin)

    if channel:

        if left and up:
            new_x[:, 0:px + s1, 0:px + s2] = xin[:, np.abs(s1):, np.abs(s2):]
            new_y[:, 0:px + s1, 0:px + s2] = yin[:, np.abs(s1):, np.abs(s2):]
        elif left and down:
            new_x[:, 0:px + s1, s2:] = xin[:, np.abs(s1):, 0:px - s2]
            new_y[:, 0:px + s1, s2:] = yin[:, np.abs(s1):, 0:px - s2]
        elif right and up:
            new_x[:, s1:, 0:px + s2] = xin[:, 0:px - s1, np.abs(s2):]
            new_y[:, s1:, 0:px + s2] = yin[:, 0:px - s1, np.abs(s2):]
        else:
            new_x[:, s1:, s2:] = xin[:, 0:px - s1, 0:px - s2]
            new_y[:, s1:, s2:] = yin[:, 0:px - s1, 0:px - s2]

    else:  # no channel dimension

        px = xin.shape[0]
        if left and up:
            new_x[0:px + s1, 0:px + s2] = xin[np.abs(s1):, np.abs(s2):]
            new_y[0:px + s1, 0:px + s2] = yin[np.abs(s1):, np.abs(s2):]
        elif left and down:
            new_x[0:px + s1, s2:] = xin[np.abs(s1):, 0:px - s2]
            new_y[0:px + s1, s2:] = yin[np.abs(s1):, 0:px - s2]
        elif right and up:
            new_x[s1:, 0:px + s2] = xin[0:px - s1, np.abs(s2):]
            new_y[s1:, 0:px + s2] = yin[0:px - s1, np.abs(s2):]
        else:
            new_x[s1:, s2:] = xin[0:px - s1, 0:px - s2]
            new_y[s1:, s2:] = yin[0:px - s1, 0:px - s2]

    if returnparams:
        return new_x, new_y, s1, s2
    else:
        return new_x, new_y


def shift_from_file(xin_fp, yin_fp, xout_path, yout_path, niter=1, channel=False):
    """ Load .npy arrays from filepaths (including name)
        and save randomly translated versions of the same arrays in corresponding
        out directories; shift parameters will be part of save filename.
    """
    xin = np.load(xin_fp)
    yin = np.load(yin_fp)

    for i in range(niter):
        xshift, yshift, s1, s2 = shift(xin, yin, shifts=(0, 0), channel=channel)
        x_fname = (xin_fp.split('/')[-1])[:-4]
        y_fname = (yin_fp.split('/')[-1])[:-4]
        np.save(f'{xout_path}/{x_fname}-trans-{s1}-{s2}', xshift)
        np.save(f'{yout_path}/{y_fname}-trans-{s1}-{s2}', yshift)
    return


def shift_single(xin, fraction=0.1, shifts=None, channel=False, returnparams=False):
    """ Generate random shifts (translations / affine transformations)
        in np arrays that represent images.
        Assumes images are square. Can have optional channel in dim 0.
    """
    assert fraction < 1.0
    px = xin.shape[1]

    if shifts is None:
        shift_max = int(fraction * px)
        shifts = np.random.randint(low=-shift_max, high=shift_max, size=(2,))
    s1, s2 = shifts
    left = s1 < 0
    up = s2 < 0
    right = not left
    down = not up

    new_x = np.zeros_like(xin)

    if channel:

        if left and up:
            new_x[:, 0:px + s1, 0:px + s2] = xin[:, np.abs(s1):, np.abs(s2):]
        elif left and down:
            new_x[:, 0:px + s1, s2:] = xin[:, np.abs(s1):, 0:px - s2]
        elif right and up:
            new_x[:, s1:, 0:px + s2] = xin[:, 0:px - s1, np.abs(s2):]
        else:
            new_x[:, s1:, s2:] = xin[:, 0:px - s1, 0:px - s2]

    else:  # no channel dimension

        px = xin.shape[0]
        if left and up:
            new_x[0:px + s1, 0:px + s2] = xin[np.abs(s1):, np.abs(s2):]
        elif left and down:
            new_x[0:px + s1, s2:] = xin[np.abs(s1):, 0:px - s2]
        elif right and up:
            new_x[s1:, 0:px + s2] = xin[0:px - s1, np.abs(s2):]
        else:
            new_x[s1:, s2:] = xin[0:px - s1, 0:px - s2]

    if returnparams:
        return new_x, s1, s2
    else:
        return new_x


# same parameters for all images
def crop(image):
    """ Cropping helper function.
        Assumes input has correct y dimensions and
        x dim >= y dim; trims excess x dim to make a square.
    """
    x0, y0 = image.size
    diff = (x0 - y0) / 2
    xl, xr = diff, x0 - diff
    ltrb = (xl, 0, xr, y0)
    return image.crop(ltrb)


def resize(image, newsize=256):
    return image.resize((newsize, newsize))


def process(image, newsize=256, grayscale=True):
    """Crops image, resizes, and converts to grayscale."""
    image = crop(image)
    image = resize(image, newsize)
    if grayscale:
        image = ImageOps.grayscale(image)
    return image


def get_closest_color(col, pal):
    return np.abs(col - pal).sum(axis=1).argmin()


def to_channel_img(img, pal):
    """ Converts an image into an image with one scalar value
        for each color in a specified palette. Each pixel's
        value is determined to be the index (in the palette)
        of the color that is closest to the pixel's original
        value.

        Note: expects image in RGBA PNG format, e.g. image
        shape should be (x_len, y_len, 4)).
    """
    img_np = np.array(img)
    img_colors = img_np[:, :, :-1]  # remove alpha channel
    s = img_np.shape
    ret_img = np.zeros(s[:-1])
    e = np.eye(pal.shape[0])

    # loop over pixels
    for x in range(s[0]):
        for y in range(s[1]):
            px_color = img_colors[x, y]
            cc = get_closest_color(px_color, pal)
            ret_img[x, y] = cc
    return ret_img


# function to process parc images into np files
def process_img(img_fp, pal, img_out_fp=None, newsize=256, shifts=None):
    """ Downsamples an image, converts it to a numpy array,
        resamples each pixel color to nearest color
        in a specified palette (pal),
        and saves the result as a numpy array.
        Arguments:
            img_fp: full filepath to an RGBA image
            pal: palette to resample to (np array of shape (nclrs, 3))
            img_out_fp: if None, output is not saved
            newsize: size to resize to (image should be square)
        Returns:
            the resulting np array of shape (newsize, newsize,)
            with values in the range 0, pal.shape[0] - 1
    """
    img = Image.open(img_fp)
    img = img.resize((newsize, newsize))
    img_channel = to_channel_img(img, pal)

    if shifts is not None:
        img_channel = shift_single(img_channel, shifts=shifts)

    # save (optionally)
    if img_out_fp is not None:
        np.save(img_out_fp, img_channel)

    return img_channel


# function to process curvature images into np files
def process_curv(img_fp, newsize=256, img_out_fp=None, shifts=None):
    """Crops image, resizes, and converts to grayscale."""
    img = Image.open(img_fp)
    img = img.resize((newsize, newsize))
    img = ImageOps.grayscale(img)
    img_np = np.array(img)

    if shifts is not None:
        img_np = shift_single(img_np, shifts=shifts)

    if img_out_fp is not None:
        np.save(img_out_fp, img_np)
    return img_np


def split(array, threshold=70, bg=255):
    T = threshold
    t = array.copy()

    t[array < T] = 2.0
    t[array >= T] = 1.0
    t[array == bg] = 0.0

    return t


def process_img_mask(img_fp, img_out_fp=None, newsize=256, shifts=None):
    """ Downsamples an image, converts it to a numpy array,
        resamples each pixel color to nearest color
        in a specified palette (pal),
        and saves the result as a numpy array.
        Arguments:
            img_fp: full filepath to an RGBA image
            pal: palette to resample to (np array of shape (nclrs, 3))
            img_out_fp: if None, output is not saved
            newsize: size to resize to (image should be square)
        Returns:
            the resulting np array of shape (newsize, newsize,)
            with values in the range 0, pal.shape[0] - 1
    """
    img = Image.open(img_fp)
    img = img.resize((newsize, newsize))
    img = np.array(img)
    img_channel = split(img)

    if shifts is not None:
        img_channel = shift_single(img_channel, shifts=shifts)

    # save (optionally)
    if img_out_fp is not None:
        np.save(img_out_fp, img_channel)

    return img_channel


### Directories ###

df_join = pd.read_csv('/scratch/groups/jyeatman/samjohns-projects/scripts/process-fns-todo.csv')
dir_data = '/scratch/groups/jyeatman/samjohns-projects/data'
dir_base = dir_data + '/atlas'

dir_in_mask = dir_base + '/curv-mask'
dir_in_curv = dir_base + '/curv'
dir_in_parc = dir_base + '/parc'

dir_out_mask = dir_base + '/curv-mask-trans-trn'
dir_out_curv = dir_base + '/curv-trans-trn'
dir_out_parc = dir_base + '/parc-trans-trn'

fns_mask = df_join['FilenameMask']
fns_curv = df_join['FilenameCurv']
fns_parc = df_join['FilenameParc']

nshifts = 5


### Main loop; generates 5 translations per image; saves outputs ###
### as pickled np arrays. (.npy) ###

for i in range(start, end):

    fnm = fns_mask[i]
    fnc = fns_curv[i]
    fnp = fns_parc[i]

    for j in range(nshifts):
        s = get_random_shift(px=256)
        if j == 0:
            s = np.array([0, 0])

        # parc
        img_fp = f'{dir_in_parc}/{fnp}'
        out_fname = f'{fnp[:-4]}-{s[0]}-{s[1]}'
        out_fp = f'{dir_out_parc}/{out_fname}'
        process_img(img_fp, pal, img_out_fp=out_fp, shifts=s)

        # curv
        img_fp = f'{dir_in_curv}/{fnc}'
        out_fname = f'{fnc[:-4]}-{s[0]}-{s[1]}'
        out_fp = f'{dir_out_curv}/{out_fname}'
        process_curv(img_fp, img_out_fp=out_fp, shifts=s)

        # curv mask
        img_fp = f'{dir_in_mask}/{fnm}'
        out_fname = f'{fnm[:-4]}-{s[0]}-{s[1]}'
        out_fp = f'{dir_out_mask}/{out_fname}'
        process_img_mask(img_fp, img_out_fp=out_fp, shifts=s)

        if i % 100 == 0:
            print(f'Processed {i} curvature images...')

print('Completed image processing .py script.')
