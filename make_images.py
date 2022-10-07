#!/usr/bin/env python
# coding: utf-8

# # Atlas Processing - Initial Image Creation
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



# image generation
for sub in subjects_todo[start:end]:
    make_subject_images(sub, angles)
    