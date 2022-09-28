#!/usr/bin/env python
# coding: utf-8

# # Atlas Processing - Initial Image Creation
# 
# ## OTS labels - Bouhali, Wiener
# 
# - Documentation: https://nilearn.github.io/auto_examples/01_plotting/plot_surf_atlas.html
# - Note: nilearn's 'surface.load_surface_data' function does not suppport .mgh / .mgz files; must use mri_convert or mris_convert (part of Freesurfer) to first convert to an acceptable format, e.g. .nii

# In[4]:

import sys
import nilearn
from nilearn import surface
from nilearn import plotting

import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
# import png  # for reloading / working with previously saved images


# ## Set up directories

# In[26]:


# === DIRECTORIES === #

# input data directories
# overall format:
# -- hbn_dir/sub-{EID}/label_subdir/parc_filename

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
out_data_dir = '/scratch/groups/jyeatman/samjohns-projects/data/atlas'
out_parc = out_data_dir + '/parc'
out_curv = out_data_dir + '/curv'
os.makedirs(out_parc, exist_ok=True)  # ensure image output directory exists
os.makedirs(out_curv, exist_ok=True)  # ensure image output directory exists
assert os.path.exists(out_curv)

# === LABELS === #

# important:
# select a subset of labels that are visible in  ventral view
# label = 43 was borderline and removed for convenience
labels_to_plot = [2, 19, 21, 23, 24, 25, 30, 37, 38, 50, 51, 57, 58, 59, 60, 61, 63, 65]


# In[5]:


fname = 'subjects-to-atlas.txt'
subjects = [s for s in os.listdir(hbn_dir) if 'sub-' in s]
subjects.sort()
with open(fname, 'w') as f:
    for s in subjects:
        f.write(s + '\n')


# In[7]:


subjects = subjects[:10]


# ## Make images

# In[27]:


curv_mask_dir = '/scratch/groups/jyeatman/samjohns-projects/data/curv-mask-jitter'
mask_fns = os.listdir(curv_mask_dir)


# In[28]:


def get_angle(s):
    angles = s.split('-')[2:4]
    angles = [float(a) for a in angles]
    return angles

def get_eid(s):
    return 'sub-' + s.split('-')[1]


# In[29]:


fn_df = pd.DataFrame({'Filename': mask_fns})
fn_df['Angles'] = fn_df['Filename'].apply(get_angle)
fn_df['EID'] = fn_df['Filename'].apply(get_eid)


# In[30]:


fn_df.sort_values('EID', inplace=True)


# In[31]:


subjects = list(fn_df['EID'].unique())


# In[34]:


def make_subject_images(sub, angles):
    parc_path = f'{hbn_dir}/{sub}{label_subdir}/{parc_filename}'
    curv_path = f'{hbn_dir}/{sub}{surf_subdir}/{curv_filename}'
    infl_path = f'{hbn_dir}/{sub}{surf_subdir}/{infl_filename}'
    
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
                               ,output_file=f'{out_parc}/{sub}-{a[0]}-{a[1]}-parc.png'   
                               # ,threshold=25.0
                               # colorbar=True
                              )

        fig, ax = plt.subplots(figsize=(8, 8))
        plotting.plot_surf_roi(infl, selected_parc
                               ,view=(a[0], a[1])
                               ,bg_map=curv
                               # ,bg_on_data=True 
                               ,figure=fig
                               ,cmap='tab20'
                               ,threshold=25.0
                               ,output_file=f'{out_curv}/{sub}-{a[0]}-{a[1]}-curv.png'
                               # colorbar=True
                              )


# In[ ]:

done_curv_fn = os.listdir(out_curv)
done_df = pd.DataFrame({'Filename': done_curv_fn})
done_df['Angles'] = done_df['Filename'].apply(get_angle)
done_df['EID'] = done_df['Filename'].apply(get_eid)

counts = done_df.groupby('EID', as_index=False).count()
counts['Count'] = counts['Angles']
del counts['Angles']
counts = counts[['EID', 'Count']]

done_subjects = list(counts[counts.Count==40]['EID'])
subjects_todo = list(set(subjects) - set(done_subjects))

# main loop: don't execute this cell unless you want to go grab a coffee
for sub in subjects_todo[start:end]:
    angles = list(fn_df[fn_df.EID==sub]['Angles'])
    make_subject_images(sub, angles)


# In[ ]:




