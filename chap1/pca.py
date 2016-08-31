#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image
from numpy import *
def pca(X):
  """ 主成分分析
   入力：X, 訓練データを平板化した配列を行として格納した行列
   出力：写像行列（次元の重要度順）, 分散, 平均 """

  # 次元数を取得
  num_data,dim = X.shape

  # データをセンタリング
  mean_X = X.mean(axis=0)
  X = X - mean_X

  if dim>num_data: 
    # PCA - 高次元のときはコンパクトな裏技を用いる
    M = dot(X,X.T) # 共分散行列
    e,EV = linalg.eigh(M) # 固有値と固有ベクトル
    tmp = dot(X.T,EV).T # ここがコンパクトな裏技
    V = tmp[::-1] # 末尾の固有ベクトルほど重要なので、反転する
    S = sqrt(e)[::-1] # 固有値の並びも反転する
    for i in range(V.shape[1]):
      V[:,i] /= S
  else:
    # PCA - 低次元なら特異値分解を用いる
    U,S,V = linalg.svd(X) 
    V = V[:num_data] # 最初のnum_dataの分だけが有用

  # 写像行列と、分散、平均を返す
  return V,S,mean_X
