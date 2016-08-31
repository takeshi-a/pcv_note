#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *

def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
  """ A. Chambolle (2005) の数式(11)記載の計算手順に基づく
    Rudin-Osher-Fatemi (ROF) ノイズ除去モデルの実装。

    入力: ノイズのある入力像（グレースケール）, Uの初期ガウス分布,
     終了判断基準の許容誤差(tolerance)、ステップ長(tau)、
     TV正規化項の重み(tv_weight)

    出力: ノイズ除去された画像,残余テクスチャ """

  m,n = im.shape # ノイズのある画像のサイズ

  # 初期化
  U = U_init 
  Px = im # 双対領域でのx成分
  Py = im # 双対領域でのy成分
  error = 1 

  while (error > tolerance): 
    Uold = U 

    # 主変数の勾配
    GradUx = roll(U,-1,axis=1)-U # Uの勾配のx成分
    GradUy = roll(U,-1,axis=0)-U # Uの勾配のy成分

    # 双対変数を更新
    PxNew = Px + (tau/tv_weight)*GradUx 
    PyNew = Py + (tau/tv_weight)*GradUy 
    NormNew = maximum(1,sqrt(PxNew**2+PyNew**2)) 

    Px = PxNew/NormNew # 双対変数のx成分を更新
    Py = PyNew/NormNew # 双対変数のy成分を更新

    # 主変数を更新
    RxPx = roll(Px,1,axis=1) # x成分の右回り変換
    RyPy = roll(Py,1,axis=0) # y成分の右回り変換

    DivP = (Px-RxPx)+(Py-RyPy) # 双対領域の発散

    U = im + tv_weight*DivP # 主変数を更新

    # 誤差を更新
    error = linalg.norm(U-Uold)/sqrt(n*m); 

  return U,im-U # ノイズ除去画像と、残余テクスチャ


