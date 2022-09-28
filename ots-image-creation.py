import nilearn
from nilearn import surface
from nilearn import plotting

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

import sys

from PIL import Image, ImageOps
import pickle
import collections

# === DIRECTORIES === #

# input data directories
# overall format:
# -- hbn_dir/sub-{EID}/label_subdir/parc_filename

assert len(sys.argv) > 1
sub_ind = int(sys.argv[1])

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

label_subdir = 'label/manual_label'
surf_subdir = 'surf'
data_dir = '/scratch/groups/jyeatman/samjohns-projects/ots-data'
subjects = [s for s in os.listdir(data_dir) if 'long' in s]
subjects.sort()

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

    sub_df = pd.DataFrame({'Subject': sub, 'Filename': label_files, 'Label': labels, 'Mesh': [surf] * len(labels),
                           'Curv': [curv] * len(labels)})
    sub_dfs.append(sub_df)

df = pd.concat(sub_dfs)

all_label_files = list(df['Filename'].unique())
all_labels = sorted([l.split('.')[1] for l in all_label_files if 'lh.' in l])

l_inc = ['OTSa', 'OTSb', 'OTSc', 'OTSd', 'OTSe']
df['LabelName'] = df['Filename'].apply(lambda l: l.split('.')[1])
df_filtered = df[df['LabelName'].isin(l_inc)]
label_to_index = dict(zip(l_inc, list(range(2, len(l_inc) + 2))))

df_filtered['LabelIndex'] = df_filtered['LabelName'].apply(lambda name: label_to_index[name])
subjects_filtered = list(df_filtered['Subject'].unique())


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

out_base_dir = '/scratch/groups/jyeatman/samjohns-projects/data'
out_parc_dir = out_base_dir + '/ots-parc-images-jitter'
out_curv_dir = out_base_dir + '/ots-curv-images-jitter'
os.makedirs(out_parc_dir, exist_ok=True)
os.makedirs(out_curv_dir, exist_ok=True)

cmap = colors.ListedColormap(['blue', 'green', 'yellow', 'red', 'purple'])
bounds = [1, 2, 3, 4, 5, 6]

def make_angles(njitter=20, scale=30, base_angle=None):
    noise = scale * (np.random.rand(njitter, 2) - 0.5)
    if base_angle is None:
        base_angle = np.array([210.0, 90.0])
    return noise + base_angle


def get_subject_labels(sub, df):
    curv = df[df['Subject'] == sub]['Curv'].iloc[0]
    stat, mesh = make_subject_stat_map(sub, df)
    return mesh, stat, curv


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
                               , view=(a[0], a[1])
                               # ,bg_map=test_curv
                               # ,bg_on_data=True
                               , figure=fig
                               , cmap=cmap
                               , output_file=f'{out_parc_path}/{sub}-{a[0]:.2f}-{a[1]:.2f}-parc.png'
                               , vmin=vmin
                               , vmax=vmax
                               # ,threshold=25.0
                               # colorbar=True
                               )

        if make_curv:
            fig, ax = plt.subplots(figsize=(8, 8))
            plotting.plot_surf_roi(infl, stat
                                   , view=(a[0], a[1])
                                   , bg_map=curv
                                   # ,bg_on_data=True
                                   , figure=fig
                                   , cmap=None
                                   , threshold=25.0
                                   , output_file=f'{out_curv_path}/{sub}-{a[0]:.2f}-{a[1]:.2f}-curv.png'
                                   # colorbar=True
                                   )

    return

sub = subjects_filtered[sub_ind]
make_ots_subject_images(sub, df_filtered,
                        out_parc_path=out_parc_dir,
                        out_curv_path=out_curv_dir,
                        cmap=cmap,
                        vmin=1.0,
                        vmax=6.0)
print(f'{sub} completed ....')