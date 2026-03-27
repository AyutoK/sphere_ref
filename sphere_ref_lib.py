import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import xarray as xr
import os
import xarray as xr
import tqdm

def set_output_dir(out="./output/"):
    """画像などの出力先ディレクトリを作成して返す。

    Parameters
    ----------
    out : str, default "./output/"
        出力先ディレクトリ。

    Returns
    -------
    output_dir : str
        作成済みの出力先ディレクトリパス。
    """
    output_dir = out  # 画像出力先
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def set_basic_params():
    """計算に用いる基本パラメータを返す。

    Notes
    -----
    - `xs` は `0..R` を `step` でサンプル。
    - `d` は入射方向ベクトル（デフォルトで -z 方向）。

    Returns
    -------
    R : float
        球半径（デフォルト 1.0）。
    step : float
        サンプリング刻み（デフォルト 0.0001）。
    xs : np.ndarray
        x サンプル配列。
    d : np.ndarray, shape (3,)
        入射方向ベクトル。
    Z0 : float
        入射開始点の z 座標。
    R2 : float
        観測球面の半径。
    """
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

def ref_rays_count(R2, xs, d, Z0, R=1.0, y=0.0, print_info=False, inc_on=False):
    """鏡面反射を仮定したレイトレーシングで、観測球面上の角度分布を返す。

    概要
    ----
    1) 入射直線と反射球の交点 p を求める
    2) 法線 n から反射方向 r を求める
    3) 反射直線と観測球の交点 p2 を求める
    4) p2 を単位化して (theta, phi) を算出

    Parameters
    ----------
    R2 : float
        観測球面の半径。
    xs : np.ndarray
        入射開始点の x 座標サンプル（p0=[x,y,Z0]）。
    d : np.ndarray, shape (3,)
        入射方向ベクトル。
    Z0 : float
        入射開始点の z 座標。
    R : float, default 1.0
        反射球半径。
    y : float, default 0.0
        入射開始点の y 座標。
    print_info : bool, default False
        サンプル数などを表示する。
    inc_on : bool, default False
        True のとき、裏側交点（入射側）も角度リストに追加する。

    Returns
    -------
    theta_arr_r2 : np.ndarray
        theta（deg）。
    phi_arr_r2 : np.ndarray
        phi（deg, 範囲 [0, 360)）。
    p0s : list[np.ndarray]
        入射開始点 p0 の列。
    p2s : list[np.ndarray]
        観測球面交点 p2 の列。
    p_hits : list[np.ndarray]
        反射点 p の列（inc_on の場合、条件により裏側交点も混在）。
    ref_dirs : list[np.ndarray]
        反射方向 r の列。

    Notes
    -----
    - inc_on=True の場合、戻り配列に裏側交点が混在しうる点に注意。
    - 掩蔽判定は実装通り（用途に応じて妥当性確認推奨）。
    """

    theta_list = []
    phi_list = []
    p0s = []
    p2s = []
    ref_dirs = []

    ts = []

    p_hits = []

    rs = []

    occl_flag = False

    for x in xs:

        occl_flag = False

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

        if inc_on:
            ti = (-b + np.sqrt(D)) / (2*a)
            if t < 0:
                continue

            pinc = p0 + ti * d  # 反射点の位置ベクトル(裏側)

            if np.linalg.norm([pinc[0],pinc[1],0]) <= R:
                pinc = np.nan # 衛星にぶつかる場合は除外(掩蔽されている)
                occl_flag = True
            else:
                thi = np.arccos(pinc[2] / np.linalg.norm(pinc))
                phii = np.arctan2(pinc[1], pinc[0])
                thi_deg = np.degrees(thi)
                phii_deg = np.degrees(phii)

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
        if inc_on and not occl_flag:
            p_hits.append(pinc)
        ref_dirs.append(r)
        theta_list.append(theta_deg)
        phi_list.append(phi_deg) # y=0では当然0 or pi
        if inc_on and not occl_flag:
            theta_list.append(thi_deg)
            phi_list.append(phii_deg)

        if inc_on:
            occl_flag = False

    theta_arr_r2 = np.array(theta_list)
    phi_arr_r2 = np.array(phi_list)

    samp = len(theta_arr_r2)

    if print_info:
        print("Total samples:", samp)

    return theta_arr_r2, phi_arr_r2, p0s, p2s, p_hits, ref_dirs

def abs_check_r2(R2, p2s):
    """点列が半径 R2 の球面上にあるか（距離誤差が閾値以下か）を検証する。

    Parameters
    ----------
    R2 : float
        期待する半径。
    p2s : array-like, shape (N,3)
        点列。

    Returns
    -------
    result_check : bool
        全点が閾値内なら True。
    """
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
    """光線（x-z断面）と theta 1Dヒストグラムを描画して保存する。

    Parameters
    ----------
    R2s : list[float]
        観測球面半径のリスト。
    xs, d, Z0, R :
        レイトレーシングのパラメータ。
    p0s, p2s, p_hits, ref_dirs :
        事前計算済みの点列・方向列（ただし本関数内で再計算も行う）。
    fp : str
        保存先ディレクトリ（末尾 "/" 推奨）。

    Outputs
    -------
    - inc_ref_rays_Rmix.png
    - theta_hist_Rmix.png

    Notes
    -----
    - 実装通り `ic` の計算が `p_hits` 長に依存するため、
      `p_hits` と `xs` の対応が崩れるケース（inc_on 等）では注意。
    """
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
    """netCDF 出力用ディレクトリを作成して返す。"""
    fp_nc = out_nc
    if not os.path.exists(fp_nc):
        os.makedirs(fp_nc)
    return fp_nc

def make_nc_sphere(R2, step, xs, d, Z0, fp_nc):
    """反射角分布を y 走査しながら計算し、netCDF として保存する。

    Parameters
    ----------
    R2 : float
        観測球面半径。
    step : float
        y 走査刻みの基準（実際の y 刻みは step*20）。
    xs, d, Z0 :
        レイトレーシングのパラメータ。
    fp_nc : str
        出力ディレクトリ。

    Outputs
    -------
    reflected_rays_R2_{R2}.nc

    Notes
    -----
    - coords の x は p_hits の x 座標。
    - x 重複がありうるため concat(join="outer") の挙動に注意。
    """

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

    if len(str(make_R2)) > 4:
        make_R2_r = make_R2.round(3)
    else:
        make_R2_r = make_R2

    ds_all.to_netcdf(fp_nc + f"reflected_rays_R2_{make_R2_r}.nc")

# ----------------------------------------
# 入射波強度を計算 ref_rays_countを流用 ※考え方が違いそう
# ----------------------------------------
def make_nc_inc(R2, step, xs, d, Z0, fp_nc):
    """入射波（と思しき）角度分布を計算し、netCDF 保存する。

    Notes
    -----
    - 実装コメント通り、反射角分布とは考え方が異なる可能性あり。
    - 保存される theta/phi は **ラジアン**（deg ではない）。
    """
    y_array = np.arange(-R2,R2+step,step*20)
    norm_R2 = R2

    for y in tqdm.tqdm(y_array):
        theta_arr_r2, phi_arr_r2, p0s, p2s, p_hits, ref_dirs = ref_rays_count(norm_R2*2, xs, d, Z0, R=norm_R2, y=y, print_info=False, inc_on=True)
        assert abs_check_r2(norm_R2, p_hits), f"R2={norm_R2}: 基準点の距離誤差が閾値を超えました。"
        p_hits_arr = np.array(p_hits)
        inc_theta_arr = np.arccos(p_hits_arr[:,2] / np.linalg.norm(p_hits, axis=1))
        inc_phi_arr = np.arctan2(p_hits_arr[:,1], p_hits_arr[:,0])
        # 0未満の場合は2πを足す
        inc_phi_arr[inc_phi_arr < 0] += 2*np.pi

        ds = xr.Dataset(
            {
                "theta": (("x"), inc_theta_arr),
                "phi": (("x"), inc_phi_arr),
            },
            coords={
                "x": np.array([x for x in range(len(p_hits))]),  # 反射点の番号(x座標ではない)
                "y": y,
            }
        )
        if y == y_array[0]:
            ds_all = ds
        else:
            ds_all = xr.concat([ds_all, ds], dim="y", join="outer")

    ds_all.to_netcdf(fp_nc + f"incident_rays_norm_{norm_R2}.nc")

# ----------------------------------------
# netCDFファイルを読み込み(角度情報)
# ----------------------------------------
def load_nc_sphere(R2, fp_nc):
    """反射角分布 netCDF（reflected_rays_R2_{R2}.nc）を読み込む。

    Raises
    ------
    FileNotFoundError
        ファイルが存在しない場合。

    Returns
    -------
    ds_all : xr.Dataset
    """
    # read_r2 = 1.2 or 2 or 5 or 10 or 1000000

    # reflected_rays_R2_{R2}.ncが存在するかチェック
    if not os.path.exists(fp_nc + f"reflected_rays_R2_{R2}.nc"):
        raise FileNotFoundError(f"{fp_nc}reflected_rays_R2_{R2}.nc does not exist")
    else:
        ds_all = xr.open_dataset(fp_nc + f"reflected_rays_R2_{R2}.nc")
        print(f"reflected_rays_R2_{R2}.nc loaded correctly.")

    return ds_all

def load_nc_inc(R2, fp_nc):
    """入射角分布 netCDF（incident_rays_norm_{R2}.nc）を読み込む。

    Raises
    ------
    FileNotFoundError
        ファイルが存在しない場合。

    Returns
    -------
    ds_all : xr.Dataset
    """
    # read_r2 = 1.2 or 2 or 5 or 10 or 1000000

    # incident_rays_norm_{R2}.ncが存在するかチェック
    if not os.path.exists(fp_nc + f"incident_rays_norm_{R2}.nc"):
        raise FileNotFoundError(f"{fp_nc}incident_rays_norm_{R2}.nc does not exist")
    else:
        ds_all = xr.open_dataset(fp_nc + f"incident_rays_norm_{R2}.nc")
        print(f"incident_rays_norm_{R2}.nc loaded correctly.")

    return ds_all

# ----------------------------------------
# 場所によって一定な入射波強度を導出
# ----------------------------------------
def calc_inc_field(R_norm, y_step=20):
    """サンプリング密度から単位立体角あたりの入射レイ密度を概算する。

    Parameters
    ----------
    R_norm : float
        スケーリング半径（観測半径など）。
    y_step : int, default 20
        y の刻み係数（ys = arange(-1,1,step*y_step)）。

    Returns
    -------
    sum_ray : int
        総レイ数。
    sum_field : float
        サンプリング面積（x-y 平面上の概算）。
    unit_inc : float
        1deg×1deg を仮定した入射密度（概算）。
    """
    R, step, xs, d, Z0, R2 = set_basic_params()
    ys = np.arange(-1,1,step*y_step)

    sum_ray = len(xs) * len(ys)
    sum_field = (xs[-1] - xs[0]) * (ys[-1] - ys[0])

    dense = sum_ray / sum_field
    sphere_unit_field = (np.radians(1) ** 2) / (1 ** 2) * (R_norm ** 2)
    unit_inc = dense * sphere_unit_field

    return sum_ray, sum_field, unit_inc

# ----------------------------------------
# 角度分布から二次元ヒートマップを作成(描画なし)
# ----------------------------------------
def make_thph_2dhist(ds_all):
    """(theta, phi) データから 2D ヒストグラムの DataArray を作成する。

    Parameters
    ----------
    ds_all : xr.Dataset
        変数 "theta" と "phi" を含む Dataset。
        角度単位は度（deg）前提。

    Returns
    -------
    heat_da : xr.DataArray
        dims=("theta","phi"), name="counts"。

    Notes
    -----
    - theta bins: 0..180（1deg刻み）
    - phi bins: 0..360（1deg刻み）
    - `make_nc_inc` 出力（rad）をそのまま入れると不整合。
    """
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
# ヒートマップをとりあえず描画(保存なし)
# ----------------------------------------
def draw_hist2d(heat_da, R2):
    """2D ヒートマップを表示する（保存はしない）。"""
    plt.figure(figsize=(10, 6))
    heat_da.plot.imshow(x="theta", y="phi", origin="lower", cmap="inferno")
    plt.xlabel("θ (deg)")
    plt.ylabel("φ (deg)")
    plt.title(f"2D histogram: counts over φ × θ (all y) R={R2}")
    plt.tight_layout()
    #plt.savefig(fp + f"2dhist_theta_vs_phi_R{R2}.png")
    plt.show()

# ----------------------------------------
# 反射率を計算するために必要な関数たち
# ----------------------------------------

# 入射ベクトルと法線ベクトルから入射角を決定する
def get_reflection_rate(d, n, e0, e1):
    """入射ベクトルと法線ベクトルから入射角を求め、Fresnel反射率を返す。

    Parameters
    ----------
    d : np.ndarray
        入射方向ベクトル。
    n : np.ndarray
        法線ベクトル（単位ベクトル想定）。
    e0 : float
        入射側比誘電率（通常 1.0）。
    e1 : float
        透過側比誘電率。

    Returns
    -------
    Rtm : float
        TM モード反射率。
    Rte : float
        TE モード反射率。
    Rave : float
        平均反射率 (|Rtm|+|Rte|)/2。
    theta_s : float
        入射角（rad）。
    """
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
    """入射角を直接与えて Fresnel 反射率を返す。

    Parameters
    ----------
    theta_s : float
        入射角（rad）。
    e0, e1 : float
        比誘電率。

    Returns
    -------
    Rtm, Rte, Rave : float
    """

    # 真空から第一層への屈折角   田中M論 式(2.9) スネルの法則
    theta_I =  np.arccos(np.sqrt(1-e0/e1*(np.sin(theta_s))**2))


    # 反射率の計算 (田中M論より)
    Rtm = np.tan(theta_s-theta_I)/np.tan(theta_s+theta_I) # TMモード
    Rte = -np.sin(theta_s-theta_I)/np.sin(theta_s+theta_I) # TEモード

    Rave = (abs(Rtm) + abs(Rte)) / 2

    return Rtm, Rte, Rave

def get_sc_angle(theta_s, R_moon, H_obs):
    """入射角から探査機が捕捉する角度を計算する。

    Parameters
    ----------
    theta_s : float
        入射角（rad）。
    R_moon : float
        天体半径。
    H_obs : float
        観測高度。

    Returns
    -------
    theta_al : float
        探査機角（rad）。
    """
    # 入射角から探査機が捕捉する角度を計算
    theta_al = 2.0*theta_s - np.arcsin(R_moon/(R_moon+H_obs)*(np.arcsin(theta_s)))

    return theta_al

# ref_rays_countの拡張版：反射率計算を追加
def ref_rays_count_ref(R2, xs, d, Z0, R, y=0.0, print_info=True, reflectance=False, e1=3.0):
    """ref_rays_count に反射率計算（任意）を追加した拡張版。

    Parameters
    ----------
    reflectance : bool|int
        `1` のとき反射率計算を実行し、amp_arr/alp_arr を返す。
    e1 : float
        比誘電率（反射率計算に使用）。

    Returns
    -------
    theta_arr_r2, phi_arr_r2, p0s, p2s, p_hits, ref_dirs
        基本のレイトレーシング結果（角度は deg）。
    amp_arr : np.ndarray
        shape (N,3) の [Rtm,Rte,Rave]（reflectance==1 のとき）。
    alp_arr : np.ndarray
        探査機角（reflectance==1 のとき）。
    """

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
    """天体（moon/ganymede）のデフォルト物理パラメータを返す。

    Parameters
    ----------
    target : str
        "moon" または "ganymede"。

    Returns
    -------
    e1, e2, H_obs, D_moon, R_moon, e1, e2, tandelta

    Notes
    -----
    - 実装通り e1/e2 を重複して返す（呼び出し側がその順で代入している）。
    """

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

# phiについて積分し、thetaについて一次元化する
def sum_phi(heat_da, phi_min, phi_max):
    """phi 範囲を切り出して phi 方向に和を取り、theta 1D にする。"""
    heat_phi_0_90 = heat_da.where((heat_da["phi"] >= phi_min) & (heat_da["phi"] <= phi_max), drop=True)
    theta_1d_counts_phi_0_90 = heat_phi_0_90.sum("phi")
    theta_1d_counts_phi_0_90.name = "counts (sum over phi 0-90)"
    return theta_1d_counts_phi_0_90

# ----------------------------------------
# 二次元ヒートマップを補正
# ----------------------------------------

# I.立体角補正(sinθで割る) --> ste_da
# II.反射率を導入 --> ste_da_ref
# III.探査機と垂直方向に補正(cosθで割る) --> ve_da

# 各補正について行うかのフラグを受け取り、最終的な分布を作成

# copilotくんに相談して改良版を作成してもらった>>steve_correction_pipeline()

def steve_correction(target, R2, fp_nc, theta_arr_r2, phi_arr_r2, p0s, p2s, p_hits, ref_dirs, amp_arr, alp_arr ,heat_da, *args):
    """補正（I:立体角, II:反射率, III:垂直方向）をフラグで適用する旧API。

    Parameters
    ----------
    args : tuple
        args[0]==1 -> I を適用
        args[1]==1 -> II を適用
        args[2]==1 -> III を適用

    Returns
    -------
    fin_da : xr.DataArray
        補正後の分布。

    Notes
    -----
    - args の要素数チェックは無く、不足すると IndexError。
    """
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
    """立体角補正として sin(theta) で割る。"""
    ste_da = heat_da / np.sin(np.radians(heat_da["theta"]))
    ste_da.name = "counts per sinθ"

    return ste_da

# ----------------------------------------
# I-I.立体角あたりに補正後、さらに単位面積あたりに補正
# (1srでR=1の球面が照らされる面積を基準)
# 立体角補正に加えて、観測点の半径R2に応じてさらにR2^2で割る
# ----------------------------------------

def i2_the_solid_angle_field(R2,heat_da):
    """立体角補正 + 単位面積補正（Rf^2 で割る）を適用する。"""
    Rf = R2 - 0.5 # 球面反射の際の焦点からの距離
    stef_da = heat_da / np.sin(np.radians(heat_da["theta"])) / (Rf ** 2)
    stef_da.name = "counts per sinθ per unit area"

    return stef_da

def i3_the_solid_angle_field_upd(R2,heat_da):
    """単位面積補正の改良版（θ依存の面積比補正を導入）。"""
    Rf, thf, thr = saf_clear_standard(R2, heat_da)

    Rf = xr.DataArray(Rf, coords={"theta": heat_da["theta"]}, dims="theta")

    SR_SfR =(Rf * thf) ** 2 / (R2 * np.radians(1)) ** 2
    SfR_Sf = 1 / (Rf ** 2)
    Sf_Sf1deg = thr

    field_cors = xr.DataArray(SR_SfR * SfR_Sf * Sf_Sf1deg, coords={"theta": heat_da["theta"]}, dims="theta")

    stef_da = heat_da / np.sin(np.radians(heat_da["theta"])) * field_cors
    stef_da.name = "counts per sinθ per unit area"
    # がっちり面積比を詰めたら中心からの面積のみ効いてくる?Rfは関係なし?

    return stef_da

# ----------------------------------------
# 改良版 : 焦点からの拡散による立体角の差異を考慮に入れる
# 基準 : 半径R=1の1度×1度範囲の面積
# ----------------------------------------

def saf_clear_standard(R2,heat_da,f=0.5):
    """i3 用の内部計算（θごとの補正係数）を作る。

    Returns
    -------
    Rfs, thfs, thrs : np.ndarray
        それぞれ焦点距離、角度幅、面積比補正。
    """
    thss = heat_da["theta"].values

    Rfs, thfs, thrs = np.zeros_like(thss), np.zeros_like(thss), np.zeros_like(thss)

    for i in range(len(thss)):

        ths = thss[i]

        Rb = np.full_like(ths, R2)
        sp, cp = np.sin(np.radians(ths+0.5)), np.cos(np.radians(ths+0.5))
        sz, cz = np.sin(np.radians(ths)), np.cos(np.radians(ths))
        sm, cm = np.sin(np.radians(ths-0.5)), np.cos(np.radians(ths-0.5))
        Rf = np.sqrt(Rb ** 2 + 0.5**2 - Rb * np.cos(np.radians(ths))) # 焦点からの距離
        Ra = (cp + np.sqrt(cp**2 + 4 * Rf**2 * f**2)) / 2
        Rc = (cm + np.sqrt(cm**2 + 4 * Rf**2 * f**2)) / 2

        r1 = np.array([Ra*cp, Ra*sp])
        r2 = np.array([Rb*cz, Rb*sz])
        r3 = np.array([Rc*cm, Rc*sm])

        sf = np.array([f,0])

        r1f = r1 - sf
        r2f = r2 - sf
        r3f = r3 - sf

        r12 = np.linalg.norm(r1f - r2f, axis=0)
        r23 = np.linalg.norm(r2f - r3f, axis=0)

        thf = (r12 + r23) / Rf

        thr = (np.radians(1) ** 2) / (thf ** 2)

        Rfs[i] = Rf
        thfs[i] = thf
        thrs[i] = thr

    return Rfs, thfs, thrs


# ----------------------------------------
# II.反射率を導入
# ----------------------------------------

# 入射角tsから探査機角度alphaを計算
def calc_alpha(ts, H, R):
    """入射角 ts（deg）から探査機角 alpha（deg）を計算する。"""
    alpha = 2 * np.radians(ts) - np.arcsin(R/(R+H) * np.sin(np.radians(ts)))
    return np.degrees(alpha)

# 目的関数：calc_alpha - alpha_target
def calc_alpha_opt(ts, H, R, alpha_target):
    """数値解法用の目的関数: calc_alpha(ts,...) - alpha_target。"""
    alpha = calc_alpha(ts, H, R)
    return alpha - alpha_target

# 探査機角度0.5度から179.5度まで1度刻みで入射角を計算し、netCDF形式で保存
def make_nc_alpha_ts(hs, fp_nc, fn = "alpha_to_theta_forhist.nc"):
    """高度配列 hs と alpha に対する入射角テーブルを作成して netCDF 保存する。"""
    match_alpha = np.arange(0.5,180,1)

    deriv_theta = np.zeros((len(match_alpha), len(hs)))

    # hs (altitudes) x match_alpha (alpha targets)
    for j, h in enumerate(hs):
        for i, ma in enumerate(match_alpha):
            d_th = optimize.fsolve(calc_alpha_opt, 1.0, args=(h-1000, 1000, ma)) # hは地表面からの距離
            deriv_theta[i, j] = float(d_th[0])

    dth_da = xr.DataArray(deriv_theta, coords={"alpha": match_alpha, "H": hs}, dims=["alpha", "H"])
    dth_da.name = "theta_s"
    dth_da

    dth_da.to_netcdf(fp_nc + fn)

# 探査機角度+高度→入射角のnetCDFデータを読み込み
def load_nc_alpha_ts(fp_nc, fn = "alpha_to_theta_forhist.nc"):
    """alpha→theta 変換テーブル（DataArray）を読み込む。"""
    # alpha_to_theta_forhist.ncが存在するかチェック
    if not os.path.exists(fp_nc + fn):
        raise FileNotFoundError(f"{fp_nc}{fn} does not exist")
    else:
        dth_da = xr.open_dataarray(fp_nc + fn)
        print(f"{fn} loaded correctly.")

    return dth_da

# II.を行う
def ii_the_reflection_rate(target, R2, ste_da, fp_nc, apply_ref="average", fn="alpha_to_theta_forhist_ver2.nc"):
    """反射率補正を適用する（alpha→theta テーブルを参照）。

    Parameters
    ----------
    target : str
        "moon" or "ganymede"。
    R2 : float
        観測半径。R2*1000 をテーブルの H と照合する（実装通り）。
    ste_da : xr.DataArray
        入力分布（例: 立体角補正後）。
    fp_nc : str
        テーブルが置いてあるディレクトリ。
    apply_ref : str
        "average" | "TE" | "TM"。
    fn : str
        変換テーブルのファイル名。

    Returns
    -------
    ste_da_ref : xr.DataArray
        反射率を乗じた分布。
    """

    R2_for = R2 * 1000 # alpha to thetaの変換で用いるRはなぜか1000倍しており、慣例化している

    att_da = load_nc_alpha_ts(fp_nc, fn=fn)

    if R2 != 1000000:
        inc_angle = att_da.sel(H=R2_for, method="nearest")
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
    """垂直方向補正として、cos を θビン平均して割り戻す。

    Notes
    -----
    - 実装通り `ve_da.fillna(0)` は代入していないため NaN が残る可能性あり。
    """
    p2s_norm = np.array(np.linalg.norm(p2s, axis=1))
    p2s_norm_true = np.full(len(p2s), p2s_norm[0])
    ths = 1e-6
    costh = np.empty(len(p2s))

    if np.all(p2s_norm - p2s_norm_true < ths):
        p2s_normed = np.array([p2s[x]/p2s_norm[0] for x in range(len(p2s))])
        #print("p2sが正常に単位ベクトル化できました")
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

def steve_correction_pipeline(heat_da, params, corrections, print_info=True):
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
        "solid_angle_field": i2_the_solid_angle_field_adapted,
        "solid_angle_field_upd": i3_the_solid_angle_field_upd_adapted,
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
        if print_info:
            print(f"✓ Applied correction: {correction_name}")
    
    return result_da


# ----------------------------------------
# 各補正関数の統一インターフェース版
# ----------------------------------------

def i_the_solid_angle_adapted(data_da, params):
    """立体角補正（パイプライン対応ラッパー）。"""
    """
    立体角補正（パイプライン対応版）
    
    既存のi_the_solid_angle関数と同じ処理を統一インターフェースで提供
    """
    return i_the_solid_angle(data_da)

def i2_the_solid_angle_field_adapted(data_da, params):
    """単位面積あたりの立体角補正（パイプライン対応ラッパー）。"""
    """
    単位面積あたりの立体角補正（パイプライン対応版）
    
    既存のi2_the_solid_angle_field関数と同じ処理を統一インターフェースで提供
    """
    return i2_the_solid_angle_field(params["R2"], data_da)

def i3_the_solid_angle_field_upd_adapted(data_da, params):
    """単位面積あたり補正（改良版）のパイプライン対応ラッパー。"""

    return i3_the_solid_angle_field_upd(params["R2"], data_da)

def ii_the_reflection_rate_adapted(data_da, params, apply_ref="average", fn="alpha_to_theta_forhist_ver2.nc"):
    """反射率補正（パイプライン対応ラッパー）。"""
    """
    反射率補正（パイプライン対応版）
    
    既存のii_the_reflection_rate関数を統一インターフェースでラップ
    """
    return ii_the_reflection_rate(
        params["target"], 
        params["R2"], 
        data_da, 
        params["fp_nc"], 
        apply_ref=apply_ref,
        fn=fn
    )


def iii_the_vertical_direction_adapted(data_da, params):
    """垂直方向補正（パイプライン対応ラッパー）。"""
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