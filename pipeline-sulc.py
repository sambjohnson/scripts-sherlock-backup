import os
import sys
import pipeline_utilities as pu
import matplotlib.pyplot as plt

start = int(sys.argv[1])
end = int(sys.argv[2])
# start, end = 0, 1  # change values during testing

hbn_dir = '/oak/stanford/groups/jyeatman/HBN/BIDS_curated/derivatives/freesurfer'
hbn_subjects = [s for s in os.listdir(hbn_dir) if s[:4]=='sub-']

save_base_dir = '/scratch/groups/jyeatman/samjohns-projects/data/atlas/sulc'
save_x_subdir = 'xs'
save_y_subdir = 'ys'
save_xdir = os.path.join(save_base_dir, save_x_subdir)
save_ydir = os.path.join(save_base_dir, save_y_subdir)

save_px2v_subdir = 'px2v'
save_pxcoord_subdir = 'pxcoord'
save_px2v_dir = os.path.join(save_base_dir, save_px2v_subdir)
save_pxcoord_dir = os.path.join(save_base_dir, save_pxcoord_subdir)

os.makedirs(save_base_dir, exist_ok=True)
os.makedirs(save_xdir, exist_ok=True)
os.makedirs(save_ydir, exist_ok=True)
os.makedirs(save_px2v_dir, exist_ok=True)
os.makedirs(save_pxcoord_dir, exist_ok=True)

parc_fn = 'lh.aparc.a2009s.annot'  # Destrieux parcellation
curv_fn = 'lh.curv'
mesh_fn = 'lh.inflated'
extra_channel_fns = ['lh.sulc']

# main loop
for sub in hbn_subjects[start:end]:

    subj_fp = os.path.join(hbn_dir, sub)
    subject_data = pu.get_freesurfer_subject_with_parc(subj_fp,
                                                    [mesh_fn], 
                                                    [curv_fn] + extra_channel_fns, 
                                                    label_files = [parc_fn])

    mesh = subject_data[mesh_fn]
    curv = subject_data[curv_fn]
    parc = subject_data[parc_fn]
    extra_channels_dict = {k: subject_data[k] for k in extra_channel_fns}

    # pipeline (below):
    # 1. create plt figures
    # 2. process (downsample, grayscale, extract channels) -> np array
    # 3. get px2v data from coordinate images

    nangles_inner = 4
    nangles_total = 12
    nangle_iterations = nangles_total // nangles_inner

    for i in range(nangle_iterations):
        fig_dict = pu.make_subject_images(mesh, 
                                          curv, 
                                          parc, 
                                          extra_channels_dict=extra_channels_dict,
                                          nangles=nangles_inner)
        np_dict = pu.process_figs(fig_dict)
        np_px_dict = pu.px2v_from_np_dict(np_dict, 
                                  mesh_coords=mesh.coordinates)
        plt.close('all')  # clear all matplotlib plots
        pu.save_subject_npys(sub, np_px_dict, save_xdir, save_ydir, 
                     extra_channel_keys=['lh.sulc'],
                     save_px2v_dir=save_px2v_dir, 
                     save_pxcoord_dir=save_pxcoord_dir)
        del np_dict
        del np_px_dict
