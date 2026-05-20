import sphere_ref_lib as srl
import sphere_ref_lib.sphere_variables as sv

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import xarray as xr

import multiprocessing as mp



def make_nc_one(R2):
    # R2ごとに独立したncを書き出す想定
    srl.make_nc_sphere(float(R2), step, xs, d, Z0, fp_nc, square=True)
    return float(R2)

if __name__ == "__main__":
    
    plt.rcParams["font.size"] = 20

    #fp = srl.set_output_dir()
    fp_nc = srl.set_output_dir_nc()
    R, step, xs, d, Z0, _ = srl.set_basic_params()
    _, _, target, apply_ref, fp, _ = sv.svv()
    e1, e2, H_obs, D_moon, R_moon, tandelta = srl.get_default_param(target)
    height = np.array([1,100,200,500,1000])

    R2s =( height + (R_moon / 1000) )/ (R_moon/1000) # ganymedeにおいて100km, 200km, 500km, 1000kmを想定

    R2s_make_nc = np.array(R2s)
    srl.make_nc_alpha_ts(R2s_make_nc*1000, fp_nc, fn = "alpha_to_theta_forhist_ver3.nc")

    plot_colors = ["red", "darkorange", "springgreen", "mediumblue", "fuchsia"]

    print("current target:", target)
    print("R2s:", R2s)


    n_proc = min(len(R2s), mp.cpu_count())
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_proc) as pool:
        results = pool.map(make_nc_one, R2s)

    print("finished:", results)