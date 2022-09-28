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
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
import pickle
import collections


# In[2]:


from nilearn import datasets

start = int(sys.argv[1])
end = int(sys.argv[2])

destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
destrieux_dict = dict(enumerate([str(l)[2:-1] for l in destrieux_atlas.labels]))


# ## 0. Set up directories

# In[3]:


# === DIRECTORIES === #

# input data directories
# overall format:
# -- hbn_dir/sub-{EID}/label_subdir/parc_filename

hbn_dir = '/scratch/users/samjohns/HBN/BIDS_curated/derivatives/freesurfer'
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


# In[4]:


label_subdir = 'label/manual_label'
surf_subdir = 'surf'
data_dir = '/scratch/groups/jyeatman/samjohns-projects/ots-data'
subjects = [s for s in os.listdir(data_dir) if 'long' in s] 
subjects.sort()


# ## Make images

# In[5]:


sub_dfs = []

for sub in subjects:
    
    curv_file = os.path.join(data_dir, sub, surf_subdir, 'lh.curv')
    infl_file = os.path.join(data_dir, sub, surf_subdir, 'lh.inflated')
    label_dir = os.path.join(data_dir, sub, label_subdir)
    label_files = [f for f in os.listdir(label_dir) if ('.label' in f) and ('lh.' in f)]

    surf = surface.load_surf_mesh(infl_file)
    curv = surface.load_surf_data(curv_file)

    labels = []
    for l in label_files:
        labels.append(surface.load_surf_data(os.path.join(label_dir, l)))
    
    sub_df = pd.DataFrame({'Subject': sub, 'Filename': label_files, 'Label': labels, 'Mesh': [surf]*len(labels), 'Curv': [curv]*len(labels)})
    sub_dfs.append(sub_df)
    
df = pd.concat(sub_dfs)


# In[6]:


all_label_files = list(df['Filename'].unique())
all_labels = sorted([l.split('.')[1] for l in all_label_files if 'lh.' in l])

l_inc = ['OTSa', 'OTSb', 'OTSc', 'OTSd', 'OTSe']
df['LabelName'] = df['Filename'].apply(lambda l: l.split('.')[1])
df_filtered = df[df['LabelName'].isin(l_inc)]
label_to_index = dict(zip(l_inc, list(range(2, len(l_inc) + 2))))


# In[7]:


df_filtered['LabelIndex'] = df_filtered['LabelName'].apply(lambda name: label_to_index[name])
subjects_filtered = list(df_filtered['Subject'].unique())


# In[8]:


df_filtered


# In[9]:


df_filtered.sort_values('Subject')


# In[10]:


def make_subject_stat_map(sub, df, return_mesh=True):

    sub_df = df[df['Subject'] == sub]

    sub_mesh = sub_df['Mesh'].iloc[0]
    sub_labs = list(sub_df['Label'])
    sub_inds = list(sub_df['LabelIndex']) 

    c = sub_mesh.coordinates.shape[0]
    m = np.ones(c)
    
    for l, i in zip(sub_labs, sub_inds):
        m[l] = float(i)
    
    if return_mesh:
        return m, sub_mesh
    else:
        return m


# In[11]:


def make_single_image(mesh, stat_map, bg=None,
                      view=[210., 90.], output_file=None, 
                      fig=None, ax=None, title=None, cmap='tab20'):
    
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(figsize=(8, 8))
    
    fig.suptitle(title)
    
    plotting.plot_surf_roi(mesh, stat_map
                           ,view=view
                           ,bg_map=bg
                           ,vmax=6.0
                           ,vmin=1.0
                           # ,bg_on_data=True
                           ,figure=fig
                           ,cmap=cmap
                           #,output_file=output_file   
                           #,threshold=20.01
                           #,colorbar=True
                          )


# ## General image-creation pipeline

# In[12]:


from matplotlib import colors
cmap = colors.ListedColormap(['blue', 'green', 'yellow', 'red', 'purple'])
bounds = [1, 2, 3, 4, 5, 6]


# In[13]:


out_base_dir = '/scratch/groups/jyeatman/samjohns-projects/data'
out_parc_dir = out_base_dir + '/ots-parc-images-jitter'
out_curv_dir = out_base_dir + '/ots-curv-images-jitter'
os.makedirs(out_parc_dir, exist_ok=True)
os.makedirs(out_curv_dir, exist_ok=True)


# In[14]:


def make_angles(njitter=20, scale=30, base_angle=None):
    noise = scale * (np.random.rand(njitter, 2) - 0.5)
    if base_angle is None:
        base_angle = np.array([210.0, 90.0])
    return noise + base_angle

def get_subject_labels(sub, df):
    curv = df[df['Subject'] == sub]['Curv'].iloc[0]
    stat, mesh = make_subject_stat_map(sub, df)
    return mesh, stat, curv

def make_ots_subject_images(sub, df, out_parc_path, out_curv_path, 
                            jitter=True, 
                            make_curv=True, 
                            cmap='tab20',
                            vmin=None, 
                            vmax=None):
        
    if jitter:
        angles = make_angles()
    else:
        angles = [[210.0, 90.0]]

    infl, stat, curv = get_subject_labels(sub, df)
    
    for a in angles:
        
        fig, ax = plt.subplots(figsize=(8, 8))
        plotting.plot_surf_roi(infl, stat
                               ,view=(a[0], a[1])
                               # ,bg_map=test_curv
                               # ,bg_on_data=True 
                               ,figure=fig
                               ,cmap=cmap
                               ,output_file=f'{out_parc_path}/{sub}-{a[0]:.2f}-{a[1]:.2f}-parc.png'
                               ,vmin=vmin
                               ,vmax=vmax
                               # ,threshold=25.0
                               # colorbar=True
                              )

        if make_curv:
            fig,  ax = plt.subplots(figsize=(8, 8))
            plotting.plot_surf_roi(infl, stat
                                   ,view=(a[0], a[1])
                                   ,bg_map=curv
                                   # ,bg_on_data=True 
                                   ,figure=fig
                                   ,cmap=None
                                   ,threshold=25.0
                                   ,output_file=f'{out_curv_path}/{sub}-{a[0]:.2f}-{a[1]:.2f}-curv.png'
                                   # colorbar=True
                                  )
        
    return


# In[15]:


# main loop: don't execute this cell unless you want to go grab a coffee
run = False

if run:
    for sub in subjects_filtered:
        make_ots_subject_images(sub, df_filtered, 
                                out_parc_path=out_parc_dir, 
                                out_curv_path=out_curv_dir, 
                                cmap=cmap, 
                                vmin=1.0, 
                                vmax=6.0)
        
        print(f'{sub} completed ....')


# In[16]:


sub = subjects_filtered[0]

sub_df = df_filtered[df_filtered.Subject == sub]
mesh = list(sub_df['Mesh'])[0]
curv = list(sub_df['Curv'])[0]
curv_mask = np.zeros_like(curv)
curv_mask[curv > 0] = 1.0


# In[17]:


# ots curv directories
ots_curv_dir = '/scratch/groups/jyeatman/samjohns-projects/data/ots-curv-images-jitter'
ots_curv_mask_dir = '/scratch/groups/jyeatman/samjohns-projects/data/ots-curv-mask-images-jitter'
os.makedirs(ots_curv_mask_dir , exist_ok=True)

# load previous angles
made_curvs = os.listdir(ots_curv_dir)


# In[18]:


made_subjects = []
made_angles = []

for m in made_curvs:
    
    msplit = m.split('-')
    angles = [float(a) for a in msplit[1:3]]
    subject = msplit[0]
    
    made_subjects.append(subject)
    made_angles.append(angles)


# In[19]:


made_df = pd.DataFrame({'Subject': made_subjects, 'Angles': made_angles})


# In[ ]:


for sub in subjects[start:end]:
    # get subject data
    sub_df = df_filtered[df_filtered.Subject == sub]
    mesh = list(sub_df['Mesh'])[0]
    curv = list(sub_df['Curv'])[0]
    # make curvature mask
    curv_mask = np.zeros_like(curv)
    curv_mask[curv > 0] = 1.0
    
    # get subject angles
    subj_angles = list(made_df[made_df['Subject']==sub]['Angles'])
    
    # make images, one per view
    for a in subj_angles:
        fig, ax = plt.subplots(figsize=(8, 8))
        plotting.plot_surf_roi(mesh, curv_mask
            ,view=a
            # ,bg_on_data=True 
            ,figure=fig
            ,cmap='gray'
            ,threshold=0.0
            ,output_file=f'{ots_curv_mask_dir}/{sub}-{a[0]:.2f}-{a[1]:.2f}-curv-mask.png'
            # ,colorbar=True
            )
