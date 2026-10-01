#%%
import numpy as np
import ultraplot as uplt
import sphere_ref_lib as srl
from sphere_ref_lib import sphere_variables as sv
import xarray as xr
import pandas as pd


tlist, fp_nc, height, ref_coords, plot_colors, cycle, sigma_thetas, sigma_phis, sigma_theta_rads, sigma_phi_rads = srl.refls.params()

drs_p, aldic = srl.refls.calc(tlist, height, ref_coords, sigma_thetas, sigma_phis)

#%%

fp = "./output_refl2/"
view_target = "moon3"
view_ref = "TM"
hmask = [1, 3, 4]
sigmask = [0, 4, 5, 6]

fn = f"fd_{view_target}_scat_ref_{view_ref}_h{len(hmask)}_scat012_swapped_v2"

srl.refls.draw(fp, view_target, view_ref, height, sigma_theta_rads, sigma_thetas, plot_colors, hmask, sigmask, drs_p, aldic, save_img=True, fn=fn)

#%%
hmask = [1]
view_ref = "TE"

view_targets = ["moon3", "moon"]

fn = f"fd_mixed_scat_ref_{view_ref}_tg{len(view_targets)}_scat012_swapped_v2"

srl.refls.draw_dual(fp, view_targets, view_ref, height, sigma_theta_rads, sigma_thetas, plot_colors, hmask, sigmask, drs_p, aldic, save_img=True, fn=fn)

#%%
# ray tracingの図をもう一回作る(カラーバー付き)

import sphere_ref_lib as srl

R, step, xs, d, Z0, R2 = srl.set_basic_params()
R2 = 2.0

theta_arr_r2, phi_arr_r2, p0s, p2s, p_hits, ref_dirs = srl.ref_rays_count(R2, xs, d, Z0, R=1.0, y=0.0, print_info=False, inc_on=False)

srl.plot_rays_hist_2d_cbar(R2, xs, d, Z0, 3.0, 2.0, True, fp="output/", R=1, save_img_rays=True)