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
view_target = "moon" 
view_ref = "TE"
hmask = [1, 3, 4]
sigmask = [0, 3, 4]

srl.refls.draw(fp, view_target, view_ref, height, sigma_theta_rads, sigma_thetas, plot_colors, hmask, sigmask, drs_p, aldic)

#%%
hmask = [1]

view_targets = ["moon3", "moon"]

srl.refls.draw_dual(fp, view_targets, view_ref, height, sigma_theta_rads, sigma_thetas, plot_colors, hmask, sigmask, drs_p, aldic, True)