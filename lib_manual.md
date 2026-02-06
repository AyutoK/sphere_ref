# sphere_ref_lib.py

推奨する略称 : srl

## srl.set_output_dir(out=./output/)
画像の出力先を指定 存在しなければ作成  

>parameters:

- out : string

>returns:

- output_dir : string

## srl.set_output_dir_nc(out_nc=./output/)
netCDF形式ファイルの出力/読み込み先を指定 存在しなければ作成  

>parameters:

- out_nc : string

>returns:

- fp_nc : string

## srl.set_basic_params()
プログラムの基本変数を設定

>returns:

- R = 1.0 : 衛星半径  
- step = 0.0001 : 照射線の刻み  
- xs : 照射するx座標 {*ndarray*}  
- d : 電波の入射方向 {*ndarray*}
- Z0 = 1000.0 : 電波源のz座標  
- R2 = 20000 : 観測点の半径  

## srl.ref_rays_count(R2, xs, d, Z0, R=1.0, y=0.0, print_info=False)
一定のy座標で光線を入射したときの反射角などを取得

>parameters:

- R2 : *float* 観測点の半径  
- xs : *ndarray* 照射するx座標  
- d : *ndarray* 電波の入射方向
- Z0 : *float* 電波源のz座標  
- R : *float* 衛星半径, optional  
- y : *float* 照射するy座標, optional  
- print_info : *bool*, optional
    - Trueにすると光線の総数をプリント

>returns:

全て*ndarray*

- theta_arr_r2 : 緯度方向の反射角(degree)
- phi_arr_r2 : 傾度方向の反射角(degree)
- p0s : 電波源の位置ベクトル
- p2s : 観測球面との交点
- p_hits : 反射点の位置ベクトル
- ref_dirs : 反射方向の方向ベクトル

## srl.abs_check_r2(R2, p2s)
全ての光線において観測球面との交点の絶対値が観測高度と一致するかテスト

>parameters:

- R2 : *float* 観測点の半径
- p2s : *ndarray* 観測球面との交点

>returns:

- result_check : *bool*
    - 閾値$10^{-6}$より差が小さければTrue

## srl.plot_rays_hist_2d(R2s, xs, d, Z0, p0s, p2s, p_hits, ref_dirs, fp, R=1.0)
光線のイメージの図示とヒストグラムの描画(複数高度)

>parameters:

- R2s : *list* 観測点の半径のリスト
- xs : *ndarray* 照射するx座標
- d : *ndarray* 電波の入射方向
- Z0 : *float* 電波源のz座標
- p0s : *ndarray* 電波源の位置ベクトル
- p2s : *ndarray* 観測球面との交点
- p_hits : *ndarray* 反射点の位置ベクトル
- ref_dirs : *ndarray* 反射方向の方向ベクトル 
- fp : *string* 画像出力先
- R : *float* 衛星半径, optional

>returns:

- 衛星と光線の反射の様子を描画したグラフ
- 角度分布のヒストグラム(カラーマップ:winter)

## srl.make_nc_sphere(R, step, xs, d, Z0, fp_nc)
球面に照射したときにおいて、各光線における反射角を計算、結果をxarrayにしてnetCDF形式で保存

>parameters:

- R2 : *float* 観測点の半径
- step : *float* 照射線の刻み
- xs : *ndarray* 照射するx座標
- d : *ndarray* 電波の入射方向
- Z0 : *float* 電波源のz座標
- fp_nc : *string* 保存場所  