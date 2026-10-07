# ------------------------------------------------------#
# fd_square_func.pyのsquare_func()で計算したxarrayを用いた計算
# 反射率、偏波、散乱関係の処理をここにまとめる
# ------------------------------------------------------#

import numpy as np
import ultraplot as uplt
import sphere_ref_lib as srl
from sphere_ref_lib import sphere_variables as sv
import xarray as xr
import pandas as pd

uplt.rc.reset()

def axis_ratio(dr_p):
    axr = dr_p.sel(ref_type="TM") / dr_p.sel(ref_type="TE")

    docp = 2 * axr / (axr **2 + 1)

    return axr, docp

def params():
    tlist = sv.target_list()
    fp_nc = srl.set_output_dir_nc()
    height = np.array([1,100,200,500,1000])

    ref_coords = ["TE", "TM", "Ave"]

    plot_colors = ["red", "darkorange", "springgreen", "mediumblue", "fuchsia"]

    cycle = uplt.Cycle(plot_colors)

    sigma_theta_rads = np.array([0, 0.1, 0.2, 0.5, 1, 2, 3, 4])
    sigma_phi_rads = np.array([0, 0.1, 0.2, 0.5, 1, 2, 3, 4])

    sigma_thetas = np.radians(sigma_theta_rads)
    sigma_phis = np.radians(sigma_phi_rads)

    return tlist, fp_nc, height, ref_coords, plot_colors, cycle, sigma_thetas, sigma_phis, sigma_theta_rads, sigma_phi_rads

def calc(tlist, height, ref_coords, sigma_thetas, sigma_phis):

    ths = np.arange(0,91,1) #入射角(0-90deg)
    ths_rad = np.radians(ths)

    # aldが都度上書きされてしまっていたため、targetごとに保存(辞書型)
    aldic = {}

    for t in tlist:
        e1, e2, H_obs, D_moon, R_moon, tandelta = srl.get_default_param(t)
        e0 = 1

        R2s = (height + (R_moon / 1000) )/ (R_moon/1000) # 100km, 200km, 500km, 1000kmを想定

        Rfd_power, ald = srl.fd.square_func(R2s, ths)
        aldic[t] = ald # targetをキーとする辞書

        rtm, rte, rave = srl.get_reflection_rate_angle(ths_rad, e0, e1)

        ref_p = np.array([Rfd_power.T * rte, Rfd_power.T * rtm, Rfd_power.T * rave])
        ref_p = np.absolute(ref_p)

        dr_p = xr.DataArray(ref_p.T, coords=[ths, height, ref_coords], dims=["theta_s", "height", "ref_type"])

        Rfdd = xr.DataArray(Rfd_power, coords=[ths, height], dims=["theta_s", "height"])
        Rfdd["ref_type"] = "None"

        dr2_p = xr.concat([dr_p,Rfdd], dim="ref_type", join="outer") #球面による効果+反射率
        
        print("current target:", t)
        #print("R2s:", R2s)
        
        if t=="moon" or t=="moon3":
            lm = 1000
        elif t=="ganymede" or t=="ganymede2" or t=="europa" or t=="calisto":
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

    return drs_p, aldic

def draw(fp: str, view_target: str, view_ref: str, height: list, sigma_theta_rads: list, sigma_thetas: list, plot_colors: list, hmask: list, sigmask: list, drs_p: xr.Dataset, aldic: dict, save_img=False, fn=""):

    view_hs = height[hmask]

    sigmasked_deg = sigma_theta_rads[sigmask]
    sigmasked = sigma_thetas[sigmask]

    h_cycle = np.array(plot_colors)[hmask]
    sls_cycle = ["-", "--", ":", "-."]

    uplt.rc.update(fontsize=21)

    lm = drs_p.sel(target=view_target).wavelength.values

    fig, ax = uplt.subplots(figsize=(12, 8))
    fig.format(suptitle=f"reflection power w/ scattering effect wavelength={lm}m ref={view_ref} <{view_target}>")

    for ii, view_sigma in enumerate(sigmasked):
        sls = sls_cycle[ii]
        view_sigma_deg = sigmasked_deg[ii]
        test_sc = drs_p.sel(target=view_target).sel(ref_type=view_ref).sel(height=view_hs).sel(sigma_theta=view_sigma)
        h_ind = hmask
        if ii == 0:
            ax.plot(
                aldic[view_target].iloc[:, h_ind],
                test_sc,
                cycle=h_cycle,
                ls=sls,
                #label=[f"{vh}km" for vh in view_hs],
                #legend="ur",
                #legend_kw={"order": "F"}
                )
        else:
            ax.plot(
                aldic[view_target].iloc[:, h_ind],
                test_sc,
                cycle=h_cycle,
                ls=sls
                )

    ax.format(xlim=(0, 140), ylim=(0, 0.3), xlabel="alpha (deg)", ylabel="reflected power / incident power")
    if save_img:
        fig.save(fp + f"{fn}.png")
    uplt.show()
    uplt.rc.reset()

"""
月:誘電率3と6の比較用
"""
def draw_dual(fp: str, view_targets: list, view_ref: str, height: list, sigma_theta_rads: list, sigma_thetas: list, plot_colors: list, hmask: list, sigmask: list, drs_p: xr.Dataset, aldic: dict, save_img=False, fn=""):

    view_hs = height[hmask]

    sigmasked_deg = sigma_theta_rads[sigmask]
    sigmasked = sigma_thetas[sigmask]

    h_cycle = np.array(plot_colors)[hmask]
    h_cycle2 = np.array(["darkorange"])

    sls_cycle = ["-", "--", ":", "-."]

    uplt.rc.update(fontsize=22)

    fig, ax = uplt.subplots(figsize=(16, 12))
    fig.format(suptitle=f"reflection power w/ scattering effect wavelength=1000m ref={view_ref} <moon ε=3 and 6>")

    for view_target in view_targets:

        lm = drs_p.sel(target=view_target).wavelength.values

        for ii, view_sigma in enumerate(sigmasked):
            sls = sls_cycle[ii]
            view_sigma_deg = sigmasked_deg[ii]
            test_sc = drs_p.sel(target=view_target).sel(ref_type=view_ref).sel(height=view_hs).sel(sigma_theta=view_sigma)
            h_ind = hmask

            draw_c = "darkorchid" if view_target=="moon3" else "limegreen"

            dualmoon_flag = (len(view_targets) == 2) and (("moon" in view_targets) and ("moon3" in view_targets))

            if dualmoon_flag:
                draw_label = "ε=3" if view_target=="moon3" else "ε=6"
            else:
                draw_label = f"{view_target}"

            if ii == 0:
                ax.plot(
                    aldic[view_target].iloc[:, h_ind],
                    test_sc,
                    c=draw_c,
                    ls=sls,
                    label=draw_label,
                    legend="ur",
                    legend_kw={"order": "F"}
                    )
            else:
                ax.plot(
                    aldic[view_target].iloc[:, h_ind],
                    test_sc,
                    c=draw_c,
                    ls=sls
                    )

    ax.format(xlim=(0, 140), ylim=(0, 0.3), xlabel="alpha (deg)", ylabel="reflected power / incident power")
    if save_img:
        fig.save(fp + f"{fn}.png")
    uplt.show()
    uplt.rc.reset()