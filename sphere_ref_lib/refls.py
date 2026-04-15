# ------------------------------------------------------#
# fd_square_func.pyのsquare_func()で計算したxarrayを用いた計算
# 反射率、偏波、散乱関係の処理をここにまとめる
# ------------------------------------------------------#

import numpy as np
import xarray as xr
import sphere_ref_lib as srl
from sphere_ref_lib import fd_square_func as fd

def axis_ratio(dr_p):
    sp_I = dr_p.sel(ref_type="TE") ** 2 + dr_p.sel(ref_type="TM") ** 2 # all electromagnetic wave strength
    sp_Q = dr_p.sel(ref_type="TE") ** 2 - dr_p.sel(ref_type="TM") ** 2

    axr = sp_Q / sp_I # All TM : -1, All TE : 1, unpolarized : 0

    return axr