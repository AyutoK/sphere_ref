# 衛星の基本情報を取得(月orガニメデ)
def get_default_param(target):
    """天体（moon/ganymede）のデフォルト物理パラメータを返す。

    Parameters
    ----------
    target : str
        "moon" "ganymede" "europa" "calisto"の4つ。
    - 2026/6/3に得率下pyファイルに移行。europa,calistoを追加。

    Returns
    -------
    e1, e2, H_obs, D_moon, R_moon, e1, e2, tandelta

    Notes
    -----
    - 実装通り e1/e2 を重複して返す（呼び出し側がその順で代入している）。
    - 2026/3/30に修正。
    """

    match target:
        case "moon":
            H_obs = 100e3       # 観測者の高度[m] (Kaguya)
            D_moon = 1*1e3      # 表層から地下構造までのレゴリスリス層(第一層)の厚さ[m]
            R_moon = 1737400.0  # 月の半径[m]
            e1 = 6.0            # レゴリス層(第一層)の比誘電率
            #e1 = 4.0
            e2 = 8.0            # レゴリス層の下の地下構造(第二層)の比誘電率
            tandelta = 0.0125   # レゴリス層(第一層)の損失角

        case "ganymede":
            H_obs = 500e3       # 観測者の高度[m] (JUICE)
            D_moon = 1*1e3      # 表層から地下構造までのレゴリスリス層(第一層)の厚さ[m]
            R_moon = 5268000.0/2.0  # ガニメデの半径[m]
            e1 = 3.0            # 第一層の比誘電率
            e2 = 87.0           # 第二層の比誘電率
            tandelta = 0.0      # 第一層の損失角

        case "europa":
            H_obs = 500e3       # 観測者の高度[m] (JUICE)
            D_moon = 1*1e3      # 表層から地下構造までのレゴリスリス層(第一層)の厚さ[m]
            R_moon = 1560.8 * 1000  # エウロパの半径[m]
            e1 = 3.0            # 第一層の比誘電率
            e2 = 87.0           # 第二層の比誘電率
            tandelta = 0.0      # 第一層の損失角

        case "calisto":
            H_obs = 500e3       # 観測者の高度[m] (JUICE)
            D_moon = 1*1e3      # 表層から地下構造までのレゴリスリス層(第一層)の厚さ[m]
            R_moon = 2410.3 * 1000  # カリストの半径[m]
            e1 = 3.0            # 第一層の比誘電率
            e2 = 87.0           # 第二層の比誘電率
            tandelta = 0.0      # 第一層の損失角

        # それ以外の時、エラーを出す
        case _:
            raise ValueError(f"Invalid target: {target}. Choose from 'moon', 'ganymede', 'europa', 'calisto'.")
    
    return e1, e2, H_obs, D_moon, R_moon, tandelta