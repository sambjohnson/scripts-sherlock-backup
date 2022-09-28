#!/usr/bin/env python
# coding: utf-8

# # Atlas Processing - Initial Image Creation
# 
# - Documentation: https://nilearn.github.io/auto_examples/01_plotting/plot_surf_atlas.html
# - Note: nilearn's 'surface.load_surface_data' function does not suppport .mgh / .mgz files; must use mri_convert or mris_convert (part of Freesurfer) to first convert to an acceptable format, e.g. .nii

# In[1]:
import sys

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

start = int(sys.argv[1])
end = int(sys.argv[2])

# hbn_dir = '/scratch/users/samjohns/HBN/BIDS_curated/derivatives/freesurfer'
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


# ## Make images

# In[5]:


def make_angles(njitter=20, scale=30, base_angle=None):
    noise = scale * (np.random.rand(njitter, 2) - 0.5)
    if base_angle is None:
        base_angle = np.array([210.0, 90.0])
    return noise + base_angle


# In[6]:


out_base_dir = '/scratch/groups/jyeatman/samjohns-projects/data'
out_parc_dir = out_base_dir + '/parc-images-jitter'
out_curv_dir = out_base_dir + '/curv-images-jitter'
os.makedirs(out_parc_dir, exist_ok=True)
os.makedirs(out_curv_dir, exist_ok=True)


# In[7]:


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


# In[8]:


# main loop: don't execute this cell unless you want to go grab a coffee
run = False
if run:
    for sub in subjects[:1]:
        make_subject_images(sub, out_parc_path=out_parc_dir, out_curv_path=out_curv_dir)


# In[9]:

	
len(subjects)


# ## 2. Create pixel2vertex maps
# 
# - assumes 3 coordinate images have already been generated

# In[10]:


# load test images
base_dir = '/scratch/groups/jyeatman/samjohns-projects/data/atlas'
tmp_img_dir = base_dir + '/tmp-coord-images'
vertex_dir = os.path.join(base_dir, 'px2vert3')
os.makedirs(vertex_dir, exist_ok=True)


# In[24]:


### utilities for translating images ### 

def shift(xin, yin, fraction=0.1, shifts=None, channel=False, returnparams=True):
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

def shift_single(xin, fraction=0.1, shifts=None, channel=False, returnparams=True):
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


# In[28]:


### utilities for creating px2vertex maps ###

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

def coords_to_vertices_old(pixel_coords, vertex_coords, return_distances=False, report_every=None):
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

def coords_to_vertices(pixel_coords, vertex_coords, return_distances=False):
    """ Loops over all vertices to match them to their closest pixels.
    """
    from sklearn.neighbors import KDTree
            
    # reshape px coordinates, remove nans
    px_shape = pixel_coords.shape[:2]
    pcs_flat = pixel_coords.reshape(-1, 3)
    pcs_flat_isnan = np.any(np.isnan(pcs_flat), axis=1)
    pcs_flat_nonan = pcs_flat[~pcs_flat_isnan]

    px_indices = np.array([(i, j) for i in range(px_shape[0]) for j in range(px_shape[1])])
    px_indices_filtered = px_indices[~pcs_flat_isnan]

    # create tree and query it
    ptree = KDTree(pcs_flat_nonan)
    dist_list, closest_pxs = ptree.query(vertex_coords)

    # reshape closest_pxs and refill nan's
    vxpx = np.take(px_indices_filtered, closest_pxs, axis=0).squeeze()

    # return results as dictionary
    # keys: (i, j) of pixels, values: list of vertices
    px2v_dict = {}
    for i in range(px_shape[0]):
        for j in range(px_shape[1]):
            px2v_dict[(i, j)] = []
    for v, px in enumerate(vxpx):
        (i, j) = px
        px2v_dict[(i, j)].append(v)
    
    if return_distances:
        return px2v_dict, dist_list
    else:
        return px2v_dict

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

def make_pixel_map(coord_img_fps, surf_coords, angle, ns=256, shift=None):
    assert len(coord_img_fps) == 3
    
    img_x = Image.open(coord_img_fps[0])
    img_y = Image.open(coord_img_fps[1])
    img_z = Image.open(coord_img_fps[2])

    img_xdown = img_x.resize((ns, ns))
    img_ydown = img_y.resize((ns, ns))
    img_zdown = img_z.resize((ns, ns))

    # note: all three color channels are equal; alpha channel irrelevant
    np_xdown = np.array(img_xdown)[:, :, 0]  
    np_ydown = np.array(img_ydown)[:, :, 0]
    np_zdown = np.array(img_zdown)[:, :, 0]
    
    # note: shift image pixels here
    if shift is not None:
        np_xdown = shift_single(np_xdown, shifts=shift, returnparams=False)
        np_ydown = shift_single(np_ydown, shifts=shift, returnparams=False)
        np_zdown = shift_single(np_zdown, shifts=shift, returnparams=False)
    
    pcs_down = get_pixel_coords(np_xdown, np_ydown, np_zdown, xrange, yrange, zrange)
    vertex_2_pxdown_dict = coords_to_vertices(pcs_down, surf_coords)
    return pcs_down, vertex_2_pxdown_dict

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

# In[29]:


import nbutils


# In[30]:


coor_df = nbutils.get_subjects_df(tmp_img_dir)

# select data with first EID for testing purposes
test_row = coor_df.loc[0]
test_eid = test_row['EID']
test_a1 = test_row['Angle1']
test_a2 = test_row['Angle2']

test_eid, test_a1, test_a2
test_infl = surface.load_surf_mesh(f'{hbn_dir}/{test_eid}/surf/lh.inflated')
test_surface_coords = test_infl.coordinates


# In[31]:


counts_df = coor_df.groupby(['EID', 'Angle1', 'Angle2'], as_index=False).count()
counts_df['Count'] = counts_df['Filename']
del counts_df['Filename']
coor_count_df = pd.merge(coor_df, counts_df, how='left', on=['EID', 'Angle1', 'Angle2'])
coor_df = coor_count_df[coor_count_df['Count'] == 3]


# In[35]:


eids = list(coor_df['EID'].unique())

for eid in eids[start:end]:  # loop over eids
    
    eid_df = coor_df[coor_df['EID'] == eid]
    a1s = list(eid_df['Angle1'])
    a2s = list(eid_df['Angle2'])
    
    eid_surf_fp = f'{hbn_dir}/{test_eid}/surf/lh.inflated'
    surf = surface.load_surf_mesh(eid_surf_fp)
    coord = surf.coordinates
    
    for a1, a2 in zip(a1s, a2s):  # loop over angles
        
        # collect the three (x, y, z) coordinate images for the current view
        view_df = eid_df[(eid_df['Angle1'] == a1) & (eid_df['Angle2'] == a2)]
        fns = sorted(list(view_df['Filename']))
        coord_img_fps = [f'{tmp_img_dir}/{fn}' for fn in fns]
        
        # make and save the px2vert maps
        pcoords, vx2coord_dict = make_pixel_map(coord_img_fps, surf_coords=coord, angle=[a1, a2], shift=None)

        pcoord_fname = f'{eid}-{a1}-{a2}-pcoord.pkl'
        vx2coord_fname = f'{eid}-{a1}-{a2}-vx2coord.pkl'

        with open(f'{vertex_dir}/{pcoord_fname}', 'wb') as f:
            pickle.dump(pcoords, f)

        with open(f'{vertex_dir}/{vx2coord_fname}', 'wb') as f:
            pickle.dump(vx2coord_dict, f)

    print(f'Completed: {eid} ...')
