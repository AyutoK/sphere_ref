# -------------------------------------- #
# refles.pyで計算したxarrayを用いた偏波を扱う計算
# あらかじめ計算結果のxarrayを用意しておく
# -------------------------------------- #

import numpy as np
import ultraplot as uplt
import sphere_ref_lib as srl
from sphere_ref_lib import sphere_variables as sv
import xarray as xr
import pandas as pd

def docp(dr_p):
    """軸比と円偏波度(DOCP)を求める

    Parameters
    ----------
    dr_p : xarray
        偏波、散乱角ごとの反射波強度をまとめたxarray。srl.reflsで計算しておく。

    Returns
    ----------
    axr : xarray
        軸比。dr_pから偏波種軸がなくなる。
    docp : xarray
        axrから計算した円偏波度。

    """
    axr = dr_p.sel(ref_type="TM") / dr_p.sel(ref_type="TE")

    docp = 2 * np.sqrt(axr) / (axr + 1)

    return axr, docp