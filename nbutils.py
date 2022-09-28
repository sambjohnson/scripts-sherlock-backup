## valid for HBN

import os
import numpy as np
import pandas as pd

def parse(st, j):
    dcs = st[:-4].split('-')
    for i in range(len(dcs)):
        if dcs[i] == '':
            dcs[i+1] = '-' + dcs[i+1]
    dcs = [d for d in dcs if d != '']
    return dcs[j]

def parse_name(st):
    return 'sub-' + parse(st, 1)

def parse_angle1(st):
    return float(parse(st, 2))

def parse_angle2(st):
    return float(parse(st, 3))

def parse_trans1(st):
    return int(parse(st, 5))

def parse_trans2(st):
    return int(parse(st, 6))

extract_dict = {'EID': parse_name, 'Angle1': parse_angle1,
                'Angle2': parse_angle2}

extract_dict_trans = {'EID': parse_name, 'Angle1': parse_angle1,
                'Angle2': parse_angle2, 'Shift1': parse_trans1, 'Shift2': parse_trans2}

extract_dict_trans_mask = {'EID': parse_name, 'Angle1': parse_angle1,
                'Angle2': parse_angle2, 'Shift1': (lambda x: int(parse(x, 6))), 'Shift2': (lambda x: int(parse(x,7)))}

def make_fn_df(fn_list: list, extract_dict: dict, fn_colname: str='Filename') -> pd.DataFrame:
    
    """ Parses each filename in a list into different fields and returns them
        as a single dataframe."""
    
    df = pd.DataFrame({fn_colname: fn_list})
    for colname, extract_fn in extract_dict.items():
        df[colname] = df[fn_colname].apply(lambda x: extract_fn(x))
    return df


def get_subjects_df(directory, extract_dict=None):
    
    import os
    import pandas as pd
    
    """ Given a directory with subject-related files,
        this function parses filenames to extract data of
        the angle and shift parameters for each subject
        and to package them into a single dataframe.
        Args:
            directory (string): where to find subject files
            extract_dict (dict): which values to extract from
                each subject file, with keys the column names in
                the returned dataframe, and values the functions
                to parse out the corresponding value.
                If None (default), this extracts only EID and Angle1, Angle2.
        Returns:
            dataframe (pandas.DataFrame) with columns"""
    
    fns = os.listdir(directory)
    
    if extract_dict is None:
        extract_dict = {'EID': parse_name, 'Angle1': parse_angle1,
            'Angle2': parse_angle2}
    
    return make_fn_df(fns, extract_dict)
