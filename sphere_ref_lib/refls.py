# ------------------------------------------------------#
# fd_square_func.pyのsquare_func()で計算したxarrayを用いた計算
# 反射率、偏波、散乱関係の処理をここにまとめる
# ------------------------------------------------------#

import numpy as np
import xarray as xr
import sphere_ref_lib as srl
from sphere_ref_lib import fd_square_func as fd

def axis_ratio(dr_p):
    axr = dr_p.sel(ref_type="TM") / dr_p.sel(ref_type="TE")

    docp = 2 * axr / (axr **2 + 1)

    return axr, docp