#%%
import numpy as np
import ultraplot as uplt
import sphere_ref_lib as srl
from sphere_ref_lib import sphere_variables as sv
import xarray as xr
import pandas as pd

#%%
fp = "./output_refl/"
fp_nc = srl.set_output_dir_nc()
R, step, xs, d, Z0, _ = srl.set_basic_params()
_, _, target, apply_ref, _, _ = sv.svv()
e1, e2, H_obs, D_moon, R_moon, tandelta = srl.get_default_param(target)
height = np.array([1,100,200,500,1000])
e0 = 1

R2s =( height + (R_moon / 1000) )/ (R_moon/1000) # ganymedeにおいて100km, 200km, 500km, 1000kmを想定
ths = np.arange(0,91,1) #入射角(0-90deg)
ths_rad = np.radians(ths)

Rfd_power, ald = srl.fd.square_func(R2s, ths)

#%%
plot_colors = ["red", "darkorange", "springgreen", "mediumblue", "fuchsia"]

cycle = uplt.Cycle(plot_colors)

fig, ax = uplt.subplots(figsize=(10,6))
ax.plot(ald, Rfd_power, cycle=cycle, label=[f"height={h}km" for h in height], legend="ur")
ax.format(xlim=(0,180), ylim=(0,1), xlabel="s/c angle alpha (deg)", ylabel="Rfd_power")
fig.format(suptitle="Rfd_power using fd_square_func")

uplt.show()
#%%
ald

#%%
rtm, rte, rave = srl.get_reflection_rate_angle(ths_rad, e0, e1)

ref_p = np.array([Rfd_power.T * rte, Rfd_power.T * rtm, Rfd_power.T * rave])
ref_p = np.absolute(ref_p)

ref_coords = ["TE", "TM", "Ave"]

dr_p = xr.DataArray(ref_p.T, coords=[ths, height, ref_coords], dims=["theta_s", "height", "ref_type"])

mapping=[[1,1,1,0,2,2,2],
		 [1,1,1,0,2,2,2],
		 [3,3,3,0,4,4,4],
		 [3,3,3,0,4,4,4],
		 [0,0,5,5,5,0,0],
		 [0,0,5,5,5,0,0]]

# heightごとに、ref_type(TE/TM/Ave)の3本線を描画
fig, ax = uplt.subplots(array=mapping,figsize=(16, 10), share=False)

for i in range(len(height)):
	h = height[i]
	for ref in ref_coords:
		y = dr_p.sel(height=h, ref_type=ref).values
		ax[i].plot(ald.iloc[:,i], y, label=ref, legend="ur")

	ax[i].format(
		xlim=(0, 180),
		ylim=(0,0.9),
		xlabel="alpha (deg)",
		ylabel="reflection rate",
		title=f"height = {h} km",
	)

fig.format(suptitle="ref_power in each height")
uplt.show()

#%%

for i in range(len(ref_coords)):
	fig, ax = uplt.subplots(figsize=(8, 5))
	ii = 0
	ref = ref_coords[i]
	for h in height:
		y = dr_p.sel(height=h, ref_type=ref).values
		ax.plot(ald.iloc[:,ii], y, label=f"height={h}km", legend="ur", cycle=cycle)
		ii += 1

	ax.format(
		xlim=(0, 140),
		ylim=(0,0.9),
		xlabel="alpha (deg)",
		ylabel="reflection rate",
	)
	fig.format(suptitle=f"ref_power in ref={ref}")

uplt.show()

#%%
# stokes parameters

sp_I = dr_p.sel(ref_type="TE") ** 2 + dr_p.sel(ref_type="TM") ** 2 # all electromagnetic wave strength
sp_Q = dr_p.sel(ref_type="TE") ** 2 - dr_p.sel(ref_type="TM") ** 2

axr_pre = sp_Q / sp_I # All TM : -1, All TE : 1, unpolarized : 0

#%%
axr = dr_p.sel(ref_type="TM") / dr_p.sel(ref_type="TE")

docp = 2 * axr / (axr **2 + 1)

#%%

fig, ax = uplt.subplots(figsize=(8,5))
fig.format(suptitle="DOCP")
ax.plot(ald, axr, cycle=cycle, label=[f"height={h}km" for h in height], legend="ur")
ax.format(xlim=(0,140), xlabel="alpha (deg)", ylabel="DOCP")
fig.save(fp + "fd_docp.png")
uplt.show()

#%%

# ------------------------------------------------------ #
# 曲面に直接散乱効果を入れようとしたもの 失敗

sigmas = [0, 5, 10, 15, 20, 25, 30]
sigma_coords = xr.DataArray(sigmas, coords=[sigmas], dims=["sigma"])

for sigma in sigmas:
	Rfd_power_i, ald_i = srl.fd.square_func_scat(R2s, ths, sigma_theta=sigma, sigma_phi=sigma)
	drs_p_i = xr.Dataset(
		{
		"Rfd_power" :(("height", "theta_s"), Rfd_power_i.T.values),
		"alpha" : (("height", "theta_s"), ald_i.T.values),
		},
		coords={
			"theta_s": ths,
			"height": height,
			"sigma": sigma,
		},)
	drs_p = drs_p_i if sigma == sigmas[0] else xr.concat([drs_p, drs_p_i], dim="sigma", join="outer")

drs_p["Rfd_power"].T.sel(height=100)

fig, ax = uplt.subplots(figsize=(8,5))
fig.format(suptitle="axis ratio (Q/I)")
ax.plot(drs_p["alpha"].sel(height=100).T, drs_p["Rfd_power"].sel(height=100).T, cycle=cycle)
ax.format(xlim=(0,140), ylim=(0,2), xlabel="alpha (deg)", ylabel="axis ratio")
#fig.save(fp + "fd_axis_ratio.png")
uplt.show()

drs_p["alpha"].sel(height=100).T

# ------------------------------------------------------ #

#%%

# ------------------------------------ #
# ここから散乱効果
sigma_theta_rads = np.array([0, 0.1, 0.2, 0.5, 1, 2, 3])
sigma_phi_rads = np.array([0, 0.1, 0.2, 0.5, 1, 2, 3])

sigma_thetas = np.radians(sigma_theta_rads)
sigma_phis = np.radians(sigma_phi_rads)

lm = 1000 # 波長(m) 想定は100MHzの電波
H = (R2s - 1) * R_moon # 探査機高度(m)
Hc = height * 1000
Fc = np.sqrt(Hc / lm) # 散乱効果に関連する数 探査機高度とFresnel半径の比の平方根

fax, sithax = np.meshgrid(Fc, sigma_thetas)
fap, siphax = np.meshgrid(Fc, sigma_phis)

st = 1 / (1 + fax * np.tan(sithax))
sp = 1 / (1 + fax * np.tan(siphax))

sce = st * sp
#scd = pd.DataFrame(sce, index=np.degrees(sigma_thetas), columns=H/1000)
scd = xr.DataArray(sce, coords=[sigma_thetas, height], dims=["sigma_theta", "height"])
scd

#%%
Rfdd = xr.DataArray(Rfd_power, coords=[ths, height], dims=["theta_s", "height"])
Rfdd["ref_type"] = "None"

dr2_p = xr.concat([dr_p,Rfdd], dim="ref_type", join="outer")
dr2_p
drs_p = dr2_p * scd

#%%
view_h = 1
view_ref = "None"

test_sc = drs_p.sel(ref_type=view_ref).sel(height=view_h)

sc_cycle = ["red", "orange", "yellow", "lime", "green", "blue", "purple"]

fig, ax = uplt.subplots(figsize=(8,5))
fig.format(suptitle=f"reflection power w/ scat effect wavelength={lm}m h={view_h}km ref={view_ref}")
ax.plot(ald.iloc[:,0], test_sc,cycle=sc_cycle, label=[f"sigma={s}deg" for s in sigma_theta_rads], legend="ur")
#ax.plot(ald, dr_p.sel(ref_type="TM").sel(height=1), cycle=cycle)
ax.format(xlim=(0,140), ylim=(0,1), xlabel="alpha (deg)", ylabel="reflection power")
fig.save(fp + f"fd_scat_ref_{view_ref}_{view_h}km.png")
uplt.show()

#%%
# 散乱位相関数っぽいので畳み込みする処理