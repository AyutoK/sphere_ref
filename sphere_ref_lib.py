import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import xarray as xr
import os
import xarray as xr
import tqdm

def set_output_dir(out="./output/"):
    output_dir = out  # 画像出力先
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def set_basic_params():
    # ----------------------------------------
    # 基本パラメータ
    # ----------------------------------------
    R = 1.0
    step = 0.0001
    xs = np.arange(0, R+step, step)

    d = np.array([0.0, 0.0, -1.0])  # 入射方向
    Z0 = 1000.0

    R2 = 20000 # 観測点の半径

    return R, step, xs, d, Z0, R2

# ----------------------------------------
# 光線追跡
# ----------------------------------------

def ref_rays_count(R2, xs, d, Z0, R=1.0, y=0.0, print_info=False):

    theta_list = []
    phi_list = []
    p0s = []
    p2s = []
    ref_dirs = []

    ts = []

    p_hits = []

    rs = []

    for x in xs:
        p0 = np.array([x, y, Z0])

        a = np.dot(d, d)
        b = 2 * np.dot(p0, d)
        c = np.dot(p0, p0) - R*R
        D = b*b - 4*a*c

        if D < 0:
            continue

        t = (-b - np.sqrt(D)) / (2*a)
        if t < 0:
            continue

        p = p0 + t * d  # 反射点の位置ベクトル
        n = p / np.linalg.norm(p)
        r = d - 2*np.dot(d, n)*n
        r /= np.linalg.norm(r) # 反射光の方向ベクトル(外側へ向かう方向を正)
        rs.append(r)

        ts.append(t)

        a2 = np.dot(r, r)
        b2 = 2 * np.dot(p, r)
        c2 = np.dot(p, p) - R2*R2
        D2 = b2*b2 - 4*a2*c2

        if D2 < 0:
            continue

        t2 = (-b2 + np.sqrt(D2)) / (2*a2)
        if t2 < 0:
            continue

        p2 = p + t2 * r # 観測球面との交点
        p2_n = p2 / np.linalg.norm(p2)  # 単位ベクトル化

        theta = np.arccos(p2_n[2])
        phi = np.arctan2(p2_n[1], p2_n[0])
        if phi < 0:
            phi += 2*np.pi

        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi)

        p0s.append(p0)
        p2s.append(p2)
        p_hits.append(p)
        ref_dirs.append(r)
        theta_list.append(theta_deg)
        phi_list.append(phi_deg) # y=0では当然0 or pi

    theta_arr_r2 = np.array(theta_list)
    phi_arr_r2 = np.array(phi_list)

    samp = len(theta_arr_r2)

    if print_info:
        print("Total samples:", samp)

    return theta_arr_r2, phi_arr_r2, p0s, p2s, p_hits, ref_dirs

def abs_check_r2(R2, p2s):
    p2s_abs = np.array(np.linalg.norm(p2s, axis=1))
    abs_true = np.full(len(p2s), R2)

    ths = 1e-6

    abs_check = p2s_abs - abs_true

    result_check = np.all(np.abs(abs_check) < ths)

    return result_check

# ----------------------------------------
# 光線の図示と角度分布
# ----------------------------------------

def plot_rays_hist_2d(R2s, xs, d, Z0, p0s, p2s, p_hits, ref_dirs, fp, R=1.0):
    # ------------------------------------------------------
    # 描画の準備
    # ------------------------------------------------------

    fig, ax = plt.subplots(figsize=(12,8))

    # 円（x-z平面）
    theta = np.linspace(0, 2*np.pi, 400)
    circle_x = R * np.cos(theta)
    circle_z = R * np.sin(theta)
    ax.plot(circle_x, circle_z, 'k', linewidth=1)

    for R2 in R2s:
        circle_x2 = R2 * np.cos(theta)
        circle_z2 = R2 * np.sin(theta)
        ax.plot(circle_x2, circle_z2, 'k', linewidth=1, linestyle="--")

    # カラーマップ
    cmap_in = plt.get_cmap("cool")
    cmap_ref = plt.get_cmap("cool")

    cmap_hist = plt.get_cmap("winter")

    # ------------------------------------------------------
    # 描画1:光線の図示
    # ------------------------------------------------------

    L = 15.0 * R  # 反射線の長さスケール
    step_plot = max(1, len(xs)//100)  # 多すぎないように間引き
    for i in range(0, len(xs), step_plot):

        ic = (i - len(p_hits)//2) / (len(p_hits)//2)

        p0 = p0s[i]
        p = p_hits[i]
        r = ref_dirs[i]
        # 入射線（p0 -> p）
        ax.plot([p0[0], p[0]], [p0[2], p[2]], color=cmap_in(int(ic * cmap_in.N)), alpha=0.8, linewidth=0.8)
        # 反射線（p -> p + L * r）
        ax.plot([p[0], p[0] + L*r[0]], [p[2], p[2] + L*r[2]], color=cmap_ref(int(ic * cmap_ref.N)), alpha=1, linewidth=0.8)

    ax.set_aspect('equal')
    ax.set_xlim(0, 10*R)
    ax.set_ylim(0, 6*R)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(f'1D (y=0) slice: incident and reflected rays on circle')
    ax.grid(True)

    plt.savefig(fp + f"inc_ref_rays_Rmix.png")
    plt.show()

    # ------------------------------------------------------
    # 描画2:角度分布
    # ------------------------------------------------------

    fig, ax = plt.subplots(2,2, figsize=(20,12))

    i = 0

    for R2 in R2s:
        theta_arr_r2, phi_arr_r2, p0s, p2s, p_hits, ref_dirs = ref_rays_count(R2, xs, d, Z0, R)
        assert abs_check_r2(R2, p2s), f"R2={R2}: 観測点の距離誤差が閾値を超えました。"

        samp = len(theta_arr_r2)

        ax[i//2, i%2].grid()
        ax[i//2, i%2].set_xticks(np.arange(0,181,10))

        ax[i//2, i%2].set_title(f"Counts 1D histogram (obs radius={R2}) (all rays={samp})")
        ax[i//2, i%2].set_xlabel("theta (deg)")
        ax[i//2, i%2].set_ylabel("Counts")
        n, bins, patches = ax[i//2, i%2].hist(theta_arr_r2, bins=180, color="blue")
        ax[i//2, i%2].set_xlim(0, 180)
        for j in range(len(patches)):
            patches[j].set_facecolor(cmap_hist(bins[j] / 180))

        i += 1

    plt.tight_layout()
    plt.savefig(fp + f"theta_hist_Rmix.png")
    plt.show()

# ----------------------------------------
# 球面において反射した角度分布を計算し、netCDF形式で保存
# ----------------------------------------
def set_output_dir_nc(out_nc="./nc_underground/"):
    fp_nc = out_nc
    if not os.path.exists(fp_nc):
        os.makedirs(fp_nc)
    return fp_nc

def make_nc_sphere(R2, step, xs, d, Z0, fp_nc):

    y_array = np.arange(-1,1,step*20)
    make_R2 = R2

    for y in tqdm.tqdm(y_array):
        theta_arr_r2, phi_arr_r2, p0s, p2s, p_hits, ref_dirs = ref_rays_count(make_R2, xs, d, Z0, R=1.0, y=y, print_info=False)
        assert abs_check_r2(make_R2, p2s), f"R2={make_R2}: 観測点の距離誤差が閾値を超えました。"
        ds = xr.Dataset(
            {
                "theta": (("x"), theta_arr_r2),
                "phi": (("x"), phi_arr_r2),
            },
            coords={
                "x": np.array([x[0] for x in p_hits]),  # 反射点のx座標
                "y": y,
            }
        )
        if y == y_array[0]:
            ds_all = ds
        else:
            ds_all = xr.concat([ds_all, ds], dim="y", join="outer")

    ds_all.to_netcdf(fp_nc + f"reflected_rays_R2_{make_R2}.nc")

# ----------------------------------------
# netCDFファイルを読み込み(角度情報)
# ----------------------------------------
def load_nc_sphere(R2, fp_nc):
    # read_r2 = 1.2 or 2 or 5 or 10 or 1000000

    # reflected_rays_R2_{R2}.ncが存在するかチェック
    if not os.path.exists(fp_nc + f"reflected_rays_R2_{R2}.nc"):
        raise FileNotFoundError(f"{fp_nc}reflected_rays_R2_{R2}.nc does not exist")
    else:
        ds_all = xr.open_dataset(fp_nc + f"reflected_rays_R2_{R2}.nc")
        print(f"reflected_rays_R2_{R2}.nc loaded correctly.")

    return ds_all

# ----------------------------------------
# 角度分布から二次元ヒートマップを作成(描画なし)
# ----------------------------------------
def make_thph_2dhist(ds_all):
    # 1) 値を 1 次元にフラット化
    theta_vals = ds_all["theta"].values.ravel()
    phi_vals = ds_all["phi"].values.ravel()

    # 2) 欠損除去
    mask = np.isfinite(theta_vals) & np.isfinite(phi_vals)
    theta_vals = theta_vals[mask]
    phi_vals = phi_vals[mask]

    # 3) ビン定義（必要に応じて解像度を変更）
    theta_bins = np.linspace(0, 180, 181)  # θ[deg]
    phi_bins = np.linspace(0, 360, 361)    # φ[deg]

    # 4) 2D ヒストグラム（x: φ, y: θ）
    H, phi_edges, theta_edges = np.histogram2d(phi_vals, theta_vals, bins=[phi_bins, theta_bins])

    # 5) 中心座標を作る
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    # 6) xarray DataArray 化（プロットしやすくするため）
    heat_da = xr.DataArray(
        H.T,  # θ を縦軸にしたいので転置
        coords={"theta": theta_centers, "phi": phi_centers},
        dims=("theta", "phi"),
        name="counts",
    )

    return heat_da

# ----------------------------------------
# 反射率を計算するために必要な関数たち
# ----------------------------------------

# 入射ベクトルと法線ベクトルから入射角を決定する
def get_reflection_rate(d, n, e0, e1):
    theta_s = np.arccos(np.dot(d,-n)) # 入射角

    # 真空から第一層への屈折角   田中M論 式(2.9) スネルの法則
    theta_I =  np.arccos(np.sqrt(1-e0/e1*(np.sin(theta_s))**2))


    # 反射率の計算 (田中M論より)
    Rtm = np.tan(theta_s-theta_I)/np.tan(theta_s+theta_I) # TMモード
    Rte = -np.sin(theta_s-theta_I)/np.sin(theta_s+theta_I) # TEモード

    Rave = (abs(Rtm) + abs(Rte)) / 2

    return Rtm, Rte, Rave, theta_s

# 入射角を直接決定する
def get_reflection_rate_angle(theta_s, e0, e1):

    # 真空から第一層への屈折角   田中M論 式(2.9) スネルの法則
    theta_I =  np.arccos(np.sqrt(1-e0/e1*(np.sin(theta_s))**2))


    # 反射率の計算 (田中M論より)
    Rtm = np.tan(theta_s-theta_I)/np.tan(theta_s+theta_I) # TMモード
    Rte = -np.sin(theta_s-theta_I)/np.sin(theta_s+theta_I) # TEモード

    Rave = (abs(Rtm) + abs(Rte)) / 2

    return Rtm, Rte, Rave

def get_sc_angle(theta_s, R_moon, H_obs):
    # 入射角から探査機が捕捉する角度を計算
    theta_al = 2.0*theta_s - np.arcsin(R_moon/(R_moon+H_obs)*(np.arcsin(theta_s)))

    return theta_al

# ref_rays_countの拡張版：反射率計算を追加
def ref_rays_count_ref(R2, xs, d, Z0, R, y=0.0, print_info=True, reflectance=False, e1=3.0):

    theta_list = []
    phi_list = []
    p0s = []
    p2s = []
    ref_dirs = []

    ts = []

    p_hits = []

    rs = []

    amps = []

    alps = []

    for x in xs:
        p0 = np.array([x, y, Z0])

        a = np.dot(d, d)
        b = 2 * np.dot(p0, d)
        c = np.dot(p0, p0) - R*R
        D = b*b - 4*a*c

        if D < 0:
            continue

        t = (-b - np.sqrt(D)) / (2*a)
        if t < 0:
            continue

        p = p0 + t * d  # 反射点の位置ベクトル
        n = p / np.linalg.norm(p)
        r = d - 2*np.dot(d, n)*n
        r /= np.linalg.norm(r) # 反射光の方向ベクトル(外側へ向かう方向を正)
        rs.append(r)

        ts.append(t)

        a2 = np.dot(r, r)
        b2 = 2 * np.dot(p, r)
        c2 = np.dot(p, p) - R2*R2
        D2 = b2*b2 - 4*a2*c2

        if D2 < 0:
            continue

        t2 = (-b2 + np.sqrt(D2)) / (2*a2)
        if t2 < 0:
            continue

        p2 = p + t2 * r # 観測球面との交点
        p2_n = p2 / np.linalg.norm(p2)  # 単位ベクトル化

        theta = np.arccos(p2_n[2])
        phi = np.arctan2(p2_n[1], p2_n[0])
        if phi < 0:
            phi += 2*np.pi

        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi)

        p0s.append(p0)
        p2s.append(p2)
        p_hits.append(p)
        ref_dirs.append(r)
        theta_list.append(theta_deg)
        phi_list.append(phi_deg) # y=0では当然0 or pi

        # ----------------------------------------
        # 反射率計算

        e0 = 1.0

        if reflectance == 1:
            Rtm, Rte, Rave, theta_s = get_reflection_rate(d,n,e0,e1)

            theta_al = get_sc_angle(theta_s, R, R2 - R)

            amps.append([Rtm,Rte,Rave])
            alps.append(theta_al)

        # ----------------------------------------

    assert len(amps) == len(alps), "リストの要素数が一致しません"

    theta_arr_r2 = np.array(theta_list)
    phi_arr_r2 = np.array(phi_list)
    amp_arr = np.array(amps)
    alp_arr = np.array(alps)

    samp = len(theta_arr_r2)

    if print_info:
        print("Total samples:", samp)

    return theta_arr_r2, phi_arr_r2, p0s, p2s, p_hits, ref_dirs, amp_arr, alp_arr

# 衛星の基本情報を取得(月orガニメデ)
def get_default_param(target):

    match target:
        case "moon":
            H_obs = 100e3       # 観測者の高度[m] (Kaguya)
            D_moon = 1*1e3      # 表層から地下構造までのレゴリスリス層(第一層)の厚さ[m]
            R_moon = 1737400.0  # 月の半径[m]
            e1 = 4.0            # レゴリス層(第一層)の比誘電率
            e2 = 8.0            # レゴリス層の下の地下構造(第二層)の比誘電率
            tandelta = 0.0125   # レゴリス層(第一層)の損失角

        case "ganymede":
            H_obs = 500e3       # 観測者の高度[m] (JUICE)
            D_moon = 1*1e3      # 表層から地下構造までのレゴリスリス層(第一層)の厚さ[m]
            R_moon = 5268000.0/2.0  # ガニメデの半径[m]
            e1 = 3.0            # 第一層の比誘電率
            e2 = 87.0           # 第二層の比誘電率
            tandelta = 0.0      # 第一層の損失角
    
    return e1, e2, H_obs, D_moon, R_moon, e1, e2, tandelta


# ----------------------------------------
# 二次元ヒートマップを補正
# ----------------------------------------

# I.立体角補正(sinθで割る) --> ste_da
# II.反射率を導入 --> ste_da_ref
# III.探査機と垂直方向に補正(cosθで割る) --> ve_da

# 各補正について行うかのフラグを受け取り、最終的な分布を作成

# copilotくんに相談して改良版を作成してもらった>>steve_correction_pipeline()

def steve_correction(target, R2, fp_nc, theta_arr_r2, phi_arr_r2, p0s, p2s, p_hits, ref_dirs, amp_arr, alp_arr ,heat_da, *args):
    n_cor = len(args)

    if args[0] == 1:
        ste_da = i_the_solid_angle(heat_da)
    else:
        ste_da = heat_da

    if args[1] == 1:
        ste_da_ref = ii_the_reflection_rate(target, R2, ste_da, fp_nc, apply_ref="average")
    else:
        ste_da_ref = ste_da

    if args[2] == 1:
        ve_da = iii_the_vertical_direction(p2s, ref_dirs, theta_arr_r2, ste_da_ref)
    else:
        ve_da = ste_da_ref



    fin_da = ve_da

    return fin_da

# ----------------------------------------
# I.立体角補正
# ----------------------------------------

def i_the_solid_angle(heat_da):
    ste_da = heat_da / np.sin(np.radians(heat_da["theta"]))
    ste_da.name = "counts per sinθ"

    return ste_da

# ----------------------------------------
# II.反射率を導入
# ----------------------------------------

# 入射角tsから探査機角度alphaを計算
def calc_alpha(ts, H, R):
    alpha = 2 * np.radians(ts) - np.arcsin(R/(R+H) * np.sin(np.radians(ts)))
    return np.degrees(alpha)

# 目的関数：calc_alpha - alpha_target
def calc_alpha_opt(ts, H, R, alpha_target):
    alpha = calc_alpha(ts, H, R)
    return alpha - alpha_target

# 探査機角度0.5度から179.5度まで1度刻みで入射角を計算し、netCDF形式で保存
def make_nc_alpha_ts(hs, fp_nc):
    match_alpha = np.arange(0.5,180,1)

    deriv_theta = np.zeros((len(match_alpha), len(hs)))

    # hs (altitudes) x match_alpha (alpha targets)
    for j, h in enumerate(hs):
        for i, ma in enumerate(match_alpha):
            d_th = optimize.fsolve(calc_alpha_opt, 1.0, args=(h, 1000, ma))
            deriv_theta[i, j] = float(d_th[0])

    dth_da = xr.DataArray(deriv_theta, coords={"alpha": match_alpha, "H": hs}, dims=["alpha", "H"])
    dth_da.name = "theta_s"
    dth_da

    dth_da.to_netcdf(fp_nc + "alpha_to_theta_forhist.nc")

# 探査機角度+高度→入射角のnetCDFデータを読み込み
def load_nc_alpha_ts(fp_nc):
    # alpha_to_theta_forhist.ncが存在するかチェック
    if not os.path.exists(fp_nc + "alpha_to_theta_forhist.nc"):
        raise FileNotFoundError(f"{fp_nc}alpha_to_theta_forhist.nc does not exist")
    else:
        dth_da = xr.open_dataarray(fp_nc + "alpha_to_theta_forhist.nc")
        print(f"alpha_to_theta_forhist.nc loaded correctly.")

    return dth_da

# II.を行う
def ii_the_reflection_rate(target, R2, ste_da, fp_nc, apply_ref="average"):

    att_da = load_nc_alpha_ts(fp_nc)

    if R2 != 1000000:
        inc_angle = att_da.sel(H=R2, method="nearest")
    else:
        inc_angle = ste_da["theta"]/2 # 無限遠では入射角 ≒ 探査機角度 / 2

    e0 = 1.0
    e1, e2, H_obs, D_moon, R_moon, e1, e2, tandelta = get_default_param(target)

    Rtm = []
    Rte = []
    Rave = []

    for i in range(len(inc_angle.values)):
        theta_s = np.radians(inc_angle.values[i])
        Rtm_p, Rte_p, Rave_p = get_reflection_rate_angle(theta_s, e0, e1)
        Rtm.append(Rtm_p)
        Rte.append(Rte_p)
        Rave.append(Rave_p)

    Rtm = np.array(Rtm)
    Rte = np.array(Rte)
    Rave = np.array(Rave)

    # ste_da の theta 軸に対して Rave をかけ合わせる（xarray の次元名でブロードキャストさせる）
    Rave_da = xr.DataArray(Rave, coords={"theta": ste_da["theta"]}, dims=("theta",))
    #Rave_da = Rave_da.where(Rave_da <= 1, 0) # 反射率1以上を0でマスク(予期しない挙動防止)

    Rte_da = xr.DataArray(Rte, coords={"theta": ste_da["theta"]}, dims=("theta",))
    #Rte_da = Rte_da.where(Rte_da <= 1, 0) # 反射率1以上を0でマスク(予期しない挙動防止)

    Rtm_da = xr.DataArray(Rtm, coords={"theta": ste_da["theta"]}, dims=("theta",))
    #Rtm_da = Rtm_da.where(Rtm_da <= 1, 0) # 反射率1以上を0でマスク(予期しない挙動防止)

    if apply_ref == "average":
        use_ref_da = Rave_da
    elif apply_ref == "TE":
        use_ref_da = Rte_da
    elif apply_ref == "TM":
        use_ref_da = Rtm_da
    else:
        raise ValueError("Invalid apply_ref value. Choose from 'average', 'TE', or 'TM'.")

    # xarray.DataArray には .abs() は無いので、Python の abs()（= DataArray.__abs__）を使う
    ste_da_ref = (ste_da * abs(use_ref_da)).rename(f"counts per sinθ with reflection rate")

    return ste_da_ref

# ----------------------------------------
# III.探査機と垂直方向に補正
# ----------------------------------------

def iii_the_vertical_direction(p2s, ref_dirs, theta_arr_r2, ste_da_ref):
    p2s_norm = np.array(np.linalg.norm(p2s, axis=1))
    p2s_norm_true = np.full(len(p2s), p2s_norm[0])
    ths = 1e-6
    costh = np.empty(len(p2s))

    if np.all(p2s_norm - p2s_norm_true < ths):
        p2s_normed = np.array([p2s[x]/p2s_norm[0] for x in range(len(p2s))])
        print("p2sが正常に単位ベクトル化できました")
        for x in range(len(p2s)):
            costh[x] = np.dot(p2s_normed[x], ref_dirs[x])
    else:
        print("p2sの距離が一定ではありません")

    ve_da = ste_da_ref.copy()

    axis_theta = []
    # 配列同士の範囲判定は「比較の連鎖」にならないように、括弧 + &（または np.logical_and）で書く
    # histのbinの範囲内にいるデータ間の平均でbinを代表させる
    for i in range(len(ve_da["theta"].values)):
        m_min = ve_da["theta"].values[i] - 0.5
        m_max = ve_da["theta"].values[i] + 0.5

        mask = np.logical_and((theta_arr_r2 < m_max),(theta_arr_r2 >= m_min))
        if np.any(mask) == False:
            ave_cos = np.nan
        else:
            ave_cos = np.average(costh[mask])

        axis_theta.append(ave_cos)

    axis_theta = np.array(axis_theta)

    axt_da = xr.DataArray(axis_theta, coords={"theta": ve_da["theta"]}, dims=("theta",))
    axt_da.name = "cos"

    ve_da = ve_da / axt_da

    ve_da.fillna(0)

    return ve_da

# ----------------------------------------
# パイプライン方式の補正関数（拡張性向上版）
# ----------------------------------------

def steve_correction_pipeline(heat_da, params, corrections):
    """
    補正関数をパイプライン形式で適用（拡張性向上版）
    
    新しい補正を追加する場合：
    1. 新しい補正関数を定義（統一インターフェース：func(data_da, params, **kwargs)）
    2. correction_functionsに登録
    3. この関数の変更は不要！
    
    Parameters
    ----------
    heat_da : xr.DataArray
        入力データ
    params : dict
        全補正関数で共有される共通パラメータ
        必須キー: target, R2, fp_nc, theta_arr_r2, phi_arr_r2, 
                 p0s, p2s, p_hits, ref_dirs, amp_arr, alp_arr
    corrections : list of tuple
        適用する補正のリスト。各要素は (補正名, パラメータ辞書) のタプル
        例: [("solid_angle", {}), 
             ("reflection_rate", {"apply_ref": "average"}),
             ("vertical_direction", {})]
    
    Returns
    -------
    result_da : xr.DataArray
        補正後のデータ
    
    Examples
    --------
    >>> params = {
    ...     "target": "moon",
    ...     "R2": 1000000,
    ...     "fp_nc": "./output/",
    ...     "theta_arr_r2": theta_arr_r2,
    ...     "phi_arr_r2": phi_arr_r2,
    ...     "p0s": p0s,
    ...     "p2s": p2s,
    ...     "p_hits": p_hits,
    ...     "ref_dirs": ref_dirs,
    ...     "amp_arr": amp_arr,
    ...     "alp_arr": alp_arr,
    ... }
    >>> corrections = [
    ...     ("solid_angle", {}),
    ...     ("reflection_rate", {"apply_ref": "average"}),
    ...     ("vertical_direction", {}),
    ... ]
    >>> result = steve_correction_pipeline(heat_da, params, corrections)
    """
    # 補正関数のマッピング（新しい補正を追加する場合はここに追加）
    correction_functions = {
        "solid_angle": i_the_solid_angle_adapted,
        "reflection_rate": ii_the_reflection_rate_adapted,
        "vertical_direction": iii_the_vertical_direction_adapted,
        # 将来的に新しい補正を追加する例：
        # "new_correction_iv": iv_new_correction_function,
    }
    
    result_da = heat_da
    
    for correction_name, correction_params in corrections:
        if correction_name not in correction_functions:
            raise ValueError(f"Unknown correction: {correction_name}. "
                           f"Available: {list(correction_functions.keys())}")
        
        # 補正関数を取得
        func = correction_functions[correction_name]
        
        # 補正を適用
        result_da = func(result_da, params, **correction_params)
        print(f"✓ Applied correction: {correction_name}")
    
    return result_da


# ----------------------------------------
# 各補正関数の統一インターフェース版
# ----------------------------------------

def i_the_solid_angle_adapted(data_da, params):
    """
    立体角補正（パイプライン対応版）
    
    既存のi_the_solid_angle関数と同じ処理を統一インターフェースで提供
    """
    return i_the_solid_angle(data_da)


def ii_the_reflection_rate_adapted(data_da, params, apply_ref="average"):
    """
    反射率補正（パイプライン対応版）
    
    既存のii_the_reflection_rate関数を統一インターフェースでラップ
    """
    return ii_the_reflection_rate(
        params["target"], 
        params["R2"], 
        data_da, 
        params["fp_nc"], 
        apply_ref=apply_ref
    )


def iii_the_vertical_direction_adapted(data_da, params):
    """
    垂直方向補正（パイプライン対応版）
    
    既存のiii_the_vertical_direction関数を統一インターフェースでラップ
    """
    return iii_the_vertical_direction(
        params["p2s"], 
        params["ref_dirs"], 
        params["theta_arr_r2"], 
        data_da
    )