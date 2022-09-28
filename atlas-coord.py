#!/usr/bin/env python
# coding: utf-8

# # Atlas Processing - Initial Image Creation
# 
# - Documentation: https://nilearn.github.io/auto_examples/01_plotting/plot_surf_atlas.html
# - Note: nilearn's 'surface.load_surface_data' function does not suppport .mgh / .mgz files; must use mri_convert or mris_convert (part of Freesurfer) to first convert to an acceptable format, e.g. .nii

# In[1]:


import nilearn
from nilearn import surface
from nilearn import plotting

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
import pickle

# import pandas as pd
# import png  # for reloading / working with previously saved images


# ## 0. Set up directories

# In[2]:


# === DIRECTORIES === #

# input data directories
# overall format:
# -- hbn_dir/sub-{EID}/label_subdir/parc_filename

# hbn_dir = '/oak/stanford/groups/jyeatman/HBN/BIDS_curated/derivatives/freesurfer'

import sys
start = int(sys.argv[1])
end = int(sys.argv[2])
hbn_dir = '/oak/stanford/groups/jyeatman/HBN/BIDS_curated/derivatives/freesurfer'
label_subdir = '/label'
surf_subdir = '/surf'

curv_filename = 'lh.curv'
infl_filename = 'lh.inflated'
pial_filename = 'lh.pial'
parc_filename = 'lh.aparc.a2009s.annot'

# output data directories
out_data_dir = '/scratch/groups/jyeatman/samjohns-projects/data'
image_out_subdir = 'parc-images'
image_out_dir = out_data_dir + '/' + image_out_subdir
image_out_dir
os.makedirs(image_out_dir, exist_ok=True)  # ensure image output directory exists
assert os.path.exists(image_out_dir)

# === LABELS === #

# important:
# select a subset of labels that are visible in  ventral view
# label = 43 was borderline and removed for convenience
labels_to_plot = [2, 19, 21, 23, 24, 25, 30, 37, 38, 50, 51, 57, 58, 59, 60, 61, 63, 65]


# In[3]:


xrange = 60.
yrange = 150.
zrange = 100.
ranges = [xrange, yrange, zrange]


# In[4]:


subjects = [s for s in os.listdir(hbn_dir) if 'sub-' in s]
subjects.sort()


# In[5]:


len(subjects)


# ## Make images

# In[6]:


def make_angles(njitter=20, scale=30, base_angle=None):
    noise = scale * (np.random.rand(njitter, 2) - 0.5)
    if base_angle is None:
        base_angle = np.array([210.0, 90.0])
    return noise + base_angle


# In[7]:


out_base_dir = '/scratch/groups/jyeatman/samjohns-projects/data'
out_parc_dir = out_base_dir + '/parc-images-jitter'
out_curv_dir = out_base_dir + '/curv-images-jitter'
os.makedirs(out_parc_dir, exist_ok=True)
os.makedirs(out_curv_dir, exist_ok=True)


# In[8]:


def make_subject_images(sub, out_parc_path, out_curv_path, jitter=True, make_curv=True):
    
    parc_path = f'{hbn_dir}/{sub}{label_subdir}/{parc_filename}'
    curv_path = f'{hbn_dir}/{sub}{surf_subdir}/{curv_filename}'
    infl_path = f'{hbn_dir}/{sub}{surf_subdir}/{infl_filename}'
    
    if jitter:
        angles = make_angles()
    else:
        angles = [[210.0, 90.0]]
    
    # check files exist
    if not (os.path.exists(parc_path) 
            and os.path.exists(curv_path) 
            and os.path.exists(infl_path)
           ):
        return

    parc = surface.load_surf_data(parc_path)
    curv = surface.load_surf_data(curv_path)
    infl = surface.load_surf_mesh(infl_path)
    
    selected_parc = np.array([labels_to_plot.index(l) if l in labels_to_plot else -1 for l in parc])
    
    for a in angles:
        fig, ax = plt.subplots(figsize=(8, 8))
        plotting.plot_surf_roi(infl, selected_parc
                               ,view=(a[0], a[1])
                               # ,bg_map=test_curv
                               # ,bg_on_data=True 
                               ,figure=fig
                               ,cmap='tab20'
                               ,output_file=f'{out_parc_path}/{sub}-{a[0]:.2f}-{a[1]:.2f}-parc.png'   
                               # ,threshold=25.0
                               # colorbar=True
                              )

        if make_curv:
            fig, ax = plt.subplots(figsize=(8, 8))
            plotting.plot_surf_roi(infl, selected_parc
                                   ,view=(a[0], a[1])
                                   ,bg_map=curv
                                   # ,bg_on_data=True 
                                   ,figure=fig
                                   ,cmap='tab20'
                                   ,threshold=25.0
                                   ,output_file=f'{out_curv_path}/{sub}-{a[0]:.2f}-{a[1]:.2f}-curv.png'
                                   # colorbar=True
                                  )
        
    return


# In[9]:


# main loop: don't execute this cell unless you want to go grab a coffee
run = False
if run:
    for sub in subjects[:1]:
        make_subject_images(sub, out_parc_path=out_parc_dir, out_curv_path=out_curv_dir)


# ## 2. Create pixel2vertex maps

# In[16]:


# load test images
img_base_dir = '/scratch/groups/jyeatman/'                 'samjohns-projects/data/parc-images-jitter-processed'
test_curv_dir = img_base_dir + '/curv-trans-val'
test_imgs = os.listdir(test_curv_dir)

vertex_map_dir = os.path.join(img_base_dir, 'vertex_maps_val')
os.makedirs(vertex_map_dir, exist_ok=True)

def parse_fname(fname, ret='angles'):
    fname_split = fname[:-4].split('-')
    for i in range(len(fname_split)-1):
        if fname_split[i] == '':
            fname_split[i+1] = '-' + fname_split[i+1]
    fname_split = [f for f in fname_split if f != '']
    angles = [float(s) for s in fname_split[2:4]]
    transl = [int(s) for s in fname_split[-2:]]
    if ret == 'transl':
        return transl
    else:
        return angles


# In[20]:


def parse_angle(fname):
    return parse_fname(fname)
    
def parse_transl(fname):
    return parse_fname(fname, ret='transl')


# In[21]:


def get_coords(arr, vrange, img_max=255.0):
    """ Convert color value (default: 0 - 255) into coordinate value.
    """
    img_max = float(img_max)  # in case img_max is supplied as int
    arr = arr.astype(dtype=np.float32)
    arr[arr==img_max] = np.nan  # set max (white background to nan)
    arr *= (2 * vrange / img_max)
    arr -= vrange
    return arr

def get_pixel_coords(np_x, np_y, np_z, xrange, yrange, zrange, img_max=255.0):
    """ Convert color into coordinate, once for each coordinate (x, y, z)
    """
    xc = get_coords(np_x, xrange)
    yc = get_coords(np_y, yrange)
    zc = get_coords(np_z, zrange)
    return np.stack([xc, yc, zc], axis=2)

def closest_coord(pixel_coords, vertex_coord, return_distance=False):
    """ Find the pixel whose (x, y, z) location is closest to that of a given vertex.
    """
    diffs = np.square(pixel_coords - vertex_coord)
    diffs = diffs.sum(axis=2)
    am = np.nanargmin(diffs)  # find the index `am` of minimum distance pixel
    dy = diffs.shape[1]
    indx = am // dy
    indy = am % dy
    if return_distance:
        return indx, indy, np.sqrt(diffs[indx, indy])  # convert `am` into 2D pixel location
    else:
        return indx, indy

def coords_to_vertices(pixel_coords, vertex_coords, return_distances=False, report_every=None):
    """ Loops over all vertices to match them to their closest pixels.
    """
    shape = pixel_coords.shape[:2]
    vertex_dir = {}
    distance_dir = {}
    for i in range(shape[0]):
        for j in range(shape[1]):
            vertex_dir[(i, j)] = []
    
    for v, vc in enumerate(vertex_coords):
        if return_distances:
            xi, yi, d = closest_coord(pixel_coords, vc, return_distance=True)
        else:
            xi, yi = closest_coord(pixel_coords, vc, return_distance=False)
        append_val = v
        if return_distances:
            append_val = [v, d]
        vertex_dir[(xi, yi)].append(append_val)
        if (report_every is not None) and (v % report_every == 0):
            print(f'Assigned {v} vertices...')
    
    return vertex_dir

def px_2_coord(vcoords, pcoords, px_vertex_dict, distance_dict=False):
    coords_vert_recon = np.zeros_like(vcoords)
    for k, v in px_vertex_dict.items():
        i, j = k  # unpack k
        coord = pcoords[i, j, :]
        for vertex in v:
            if distance_dict:
                vertex = vertex[0]
            coords_vert_recon[vertex] = coord
    return coords_vert_recon


# In[24]:

test_df = pd.read_csv('/scratch/groups/jyeatman/samjohns-projects/scripts/coord-todo.csv')
tmp_img_dir = '/scratch/groups/jyeatman/samjohns-projects/data/atlas/tmp-coord-images'

subjects = list(test_df['EID'])
angle1s = list(test_df['Angle1'])
angle2s = list(test_df['Angle2'])

# In[25]:


def make_subject_images(subdir, savedir, save_prefix, a=[210., 0.]):
    """ Make subject image to record x, y, and z of vertices.
        Saved as 3 grayscale images.
    """
    xrange = 60.
    yrange = 150.
    zrange = 100.
    ranges = [xrange, yrange, zrange]
    
    curv_path = f'{subdir}/surf/lh.curv'
    infl_path = f'{subdir}/surf/lh.inflated'
    curv = surface.load_surf_data(curv_path)
    infl = surface.load_surf_mesh(infl_path)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(3):
        
        plotting.plot_surf_roi(infl, infl.coordinates[:, i]
                               ,view=(a[0], a[1])
                               # ,bg_map=test_curv
                               # ,bg_on_data=True 
                               ,figure=fig
                               ,cmap='gray'
                               ,output_file=f'{savedir}/{save_prefix}-{a[0]:.2f}-{a[1]:.2f}-coord-{i}.png'
                               ,vmin=-ranges[i]
                               ,vmax=ranges[i]
                               # ,threshold=25.0
                               # colorbar=True
                              )


# # Make coord images here

make_images = True
if make_images:
    for i, (sub, a1, a2) in enumerate(zip(subjects[start:end], angle1s[start:end], angle2s[start:end])):
        subdir = f'{hbn_dir}/{sub}'
        save_prefix = sub
        if os.path.exists(subdir):
            make_subject_images(subdir, tmp_img_dir, save_prefix, a=[a1, a2])
        if i % 10 == 0:
            print(f'Completed {i} subjects....')
