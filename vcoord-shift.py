import nbutils
import os
import sys
import pandas as pd
import pickle

start = int(sys.argv[1])
end = int(sys.argv[2])


def get_shifts(s):
    sl = (s[:-4]).split('-')
    for i in range(len(sl) - 1):
        if sl[i] == '':
            sl[i + 1] = '-' + sl[i + 1]
    sl = [s for s in sl if s != '']
    s1 = int(sl[-2])
    s2 = int(sl[-1])
    return s1, s2


def apply_shift_to_vdict(vdict, s1, s2, max_ind=255):
    """
        Takes a dictionary of (i, j) pixel indices to
        vertices, and translates (i -> i + s1, j -> j+ s2)
        according to a shift (s1, s2).

    """
    s1, s2 = int(s1), int(s2)
    vdict_shifted = {k: [] for k in vdict.keys()}
    for k, v in vdict.items():
        if len(v) > 0:
            i, j = k  # unpack i, j pixel coordinates of key
            i_s, j_s = i + s1, j + s2
            if i_s < 0 or j_s < 0 or i_s > max_ind or j_s > max_ind:
                continue
            vdict_shifted[(i_s, j_s)] = v
    return vdict_shifted


atlas_dir = '/scratch/groups/jyeatman/samjohns-projects/data/atlas'
px2vert_dir = atlas_dir + '/px2vert'
curv_dir = atlas_dir + '/curv-trans-trn'
vcoord_shift_dir = atlas_dir + '/vcoord'

vert_df = nbutils.get_subjects_df(px2vert_dir)
vert_df['Type'] = vert_df['Filename'].apply(lambda f: 'V' if 'vx2coord' in f else 'P')

curv_df = nbutils.get_subjects_df(curv_dir)
curv_df['Shift1'] = curv_df['Filename'].apply(lambda fn: get_shifts(fn)[0])
curv_df['Shift2'] = curv_df['Filename'].apply(lambda fn: get_shifts(fn)[1])

shift_df = curv_df[['EID', 'Angle1', 'Angle2', 'Shift1', 'Shift2']]
to_shift_df = pd.merge(vert_df[vert_df['Type']=='V'], shift_df, how='left', on=['EID', 'Angle1', 'Angle2'])
to_shift_df = to_shift_df.dropna()
to_shift_df = to_shift_df.astype({'Shift1': int, 'Shift2': int})

shift_fns = list(to_shift_df['Filename'].unique())

for v_fn in shift_fns[start:end]:

    fn_df = to_shift_df[to_shift_df['Filename'] == v_fn]

    with open(f'{px2vert_dir}/{v_fn}', 'rb') as f:
        vfile = pickle.load(f)

    shift1s = list(fn_df['Shift1'])
    shift2s = list(fn_df['Shift2'])

    for s1, s2 in zip(shift1s, shift2s):
        v_shifted = apply_shift_to_vdict(vfile, s1, s2)
        vx2coord_fname = f'{v_fn[:-13]}-{s1}-{s2}-vx2coord.pkl'
        vx2coord_fpath = vcoord_shift_dir + '/' + vx2coord_fname

        with open(vx2coord_fpath, 'wb') as f:
            pickle.dump(v_shifted, f)
