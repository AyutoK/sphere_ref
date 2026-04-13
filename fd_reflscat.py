#%%
import numpy as np
import ultraplot as uplt
import sphere_ref_lib as srl
import sphere_variables as sv
import fd_square_func as fd
import xarray as xr

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

#%%
Rfd_power, ald = fd.square_func(R2s, ths)

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

axr = sp_Q / sp_I # All TM : -1, All TE : 1, unpolarized : 0

#%%

fig, ax = uplt.subplots(figsize=(8,5))
fig.format(suptitle="axis ratio (Q/I)")
ax.plot(ald, axr, cycle=cycle, label=[f"height={h}km" for h in height], legend="ur")
ax.format(xlim=(0,140), xlabel="alpha (deg)", ylabel="axis ratio")
fig.save(fp + "fd_axis_ratio.png")
uplt.show()