# fd_reflscat.pyのxarray作成コードを取り出して改良したもの
# 実装でき次第関数化してsrl.reflsに移す予定

#%%
import numpy as np
import ultraplot as uplt
import sphere_ref_lib as srl
from sphere_ref_lib import sphere_variables as sv
import xarray as xr
import pandas as pd

uplt.rc.reset()

#%%
fp = "./output_refl2/"
fp_nc = srl.set_output_dir_nc()
height = np.array([1,100,200,500,1000])

ref_coords = ["TE", "TM", "Ave"]

plot_colors = ["red", "darkorange", "springgreen", "mediumblue", "fuchsia"]

cycle = uplt.Cycle(plot_colors)

ths = np.arange(0,91,1) #入射角(0-90deg)
ths_rad = np.radians(ths)

sigma_theta_rads = np.array([0, 0.1, 0.2, 0.5, 1, 2, 3])
sigma_phi_rads = np.array([0, 0.1, 0.2, 0.5, 1, 2, 3])

sigma_thetas = np.radians(sigma_theta_rads)
sigma_phis = np.radians(sigma_phi_rads)

tlist = sv.target_list()

for t in tlist:
    e1, e2, H_obs, D_moon, R_moon, tandelta = srl.get_default_param(t)
    e0 = 1

    R2s = (height + (R_moon / 1000) )/ (R_moon/1000) # 100km, 200km, 500km, 1000kmを想定

    Rfd_power, ald = srl.fd.square_func(R2s, ths)
	
    rtm, rte, rave = srl.get_reflection_rate_angle(ths_rad, e0, e1)

    ref_p = np.array([Rfd_power.T * rte, Rfd_power.T * rtm, Rfd_power.T * rave])
    ref_p = np.absolute(ref_p)

    dr_p = xr.DataArray(ref_p.T, coords=[ths, height, ref_coords], dims=["theta_s", "height", "ref_type"])

    Rfdd = xr.DataArray(Rfd_power, coords=[ths, height], dims=["theta_s", "height"])
    Rfdd["ref_type"] = "None"

    dr2_p = xr.concat([dr_p,Rfdd], dim="ref_type", join="outer") #球面による効果+反射率
    
    print("current target:", t)
    print("R2s:", R2s)
	
    if t=="moon":
        lm = 1000
    elif t=="ganymede" or t=="europa" or t=="calisto":
        lm = 100 # 波長(m) 想定は100MHzの電波

    H = (R2s - 1) * R_moon # 探査機高度(m)
    Hc = height * 1000
    Fc = np.sqrt(Hc / lm) # 散乱効果に関連する数 探査機高度とFresnel半径の比の平方根

    fax, sithax = np.meshgrid(Fc, sigma_thetas)
    fap, siphax = np.meshgrid(Fc, sigma_phis)

    st = 1 / (1 + fax * np.tan(sithax))
    sp = 1 / (1 + fax * np.tan(siphax))

    sce = st * sp
    scd = xr.DataArray(sce, coords=[sigma_thetas, height], dims=["sigma_theta", "height"]) #散乱による効果

    drs_p_bc = (dr2_p * scd).expand_dims(target=[t]) #target=tにおける反射波の強度
    drs_p_bc = drs_p_bc.assign_coords(wavelength=("target", [lm])) #targetごとに波長を保存

    drs_p = drs_p_bc if t == tlist[0] else xr.concat([drs_p, drs_p_bc], dim="target", join="outer", coords="minimal")

drs_p

#%%
# 高度と散乱角で、色と線種を入れ替えた版
view_target = "moon"
hmask = [1,3,4] # [1km,100km,200km,500km,1000km]のうち、どの高度を表示するか
view_hs = height[hmask]
view_ref = "Ave"

sigmask = [0, 3, 4]
sigmasked_deg = sigma_theta_rads[sigmask]
sigmasked = sigma_thetas[sigmask]

h_cycle = np.array(plot_colors)[hmask]
sls_cycle = ["-", "--", ":"]

uplt.rc.update(fontsize=13)

fig, ax = uplt.subplots(figsize=(10, 8))
fig.format(suptitle=f"reflection power w/ scat effect wavelength={lm}m ref={view_ref} <{view_target}>")

for ii, view_sigma in enumerate(sigmasked):
	sls = sls_cycle[ii]
	view_sigma_deg = sigmasked_deg[ii]
	test_sc = drs_p.sel(target=view_target).sel(ref_type=view_ref).sel(height=view_hs).sel(sigma_theta=view_sigma)
	h_ind = hmask
	ax.plot(
		ald.iloc[:, h_ind],
		test_sc,
		cycle=h_cycle,
		ls=sls,
		label=[f"{vh}km, sigma={view_sigma_deg}deg" for vh in view_hs],
		legend="b",
		legend_kw={"order": "F"}
		)

ax.format(xlim=(0, 140), ylim=(0, 0.4), xlabel="alpha (deg)", ylabel="reflection power")
fig.save(fp + f"fd_{view_target}_scat_ref_{view_ref}_variation_h{len(view_hs)}_swapped.png")
uplt.show()
uplt.rc.reset()

#%%
Rfdd.values

#%%
drs_p.sel(target="moon").sel(ref_type="Ave").sel(height=100).sel(sigma_theta=0)

#%%
