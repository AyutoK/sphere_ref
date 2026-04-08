# 木星天頂角と探査機の高さから、木星電波の表面反射波の強度を算出する関数
# 反射率、散乱などは考慮せず、球面反射による空間的な拡散のみ考慮

# source:fd_square.ipynb

# 書き始めた日: 2026/04/07

import numpy as np
import pandas as pd
import sphere_ref_lib as srl

def fd_square_func(R2s, ths):
    """
    木星天頂角と探査機の高さから、木星電波の表面反射波の強度を導出
    Parameters
    ----------
    R2s : list
        探査機の高さのリスト。単位は木星半径。
    ths : list
        木星天頂角のリスト。単位は度。
    Returns
    -------
    Rfd_power : pandas.DataFrame
        木星電波の表面反射波強度の直達波に対する比。行が木星天頂角、列が探査機の高さに対応。
    ald : pandas.DataFrame
        木星天頂角と探査機の高さから算出される探査機角度αの値。行が木星天頂角、列が探査機の高さに対応。
    """

    ths_rad = np.radians(ths)
    Rax, thax = np.meshgrid(R2s, ths_rad)

    alpha_deg = srl.calc_alpha(np.degrees(thax),(Rax-1),1)
    ald = pd.DataFrame(alpha_deg, index=ths, columns=["R=" + str(x)  for x in R2s])
    alpha_rad = np.radians(alpha_deg)

    focus_p = 1/2 * np.cos(thax)
    focus_p2 = 1/(2 * np.cos(thax))

    theta_d = alpha_rad - thax

    L = np.sqrt(Rax ** 2 + 1 ** 2 - 2 * 1 * Rax * np.cos(theta_d))

    Rf = L + focus_p
    Rg = np.sqrt(focus_p2 ** 2 + 1 ** 2 -2 * focus_p2 * 1 * np.cos(thax))

    Rfd = pd.DataFrame(Rf, index=ths, columns=["R=" + str(x)  for x in R2s])
    Rfd_norm = focus_p

    gt = Rfd_norm / Rfd
    gp = Rg / (L + Rg)

    Rfd_power = gt * gp

    return Rfd_power, ald