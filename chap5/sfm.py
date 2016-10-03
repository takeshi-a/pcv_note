#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image
from numpy import *
from scipy import linalg
from pylab import *

def compute_fundamental(x1,x2):
  """ 正規化8点法を使って対応点群(x1,x2:3*nの配列)
      から基礎行列を計算する。各列は次のような並びである。
      [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

  n = x1.shape[1]
  if x2.shape[1] != n:
    raise ValueError("Number of points don't match.")

  # 方程式の行列を作成する
  A = zeros((n,9))
  for i in range(n):
    A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
        x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
        x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]

  # 線形最小2乗法で計算する
  U,S,V = linalg.svd(A)
  F = V[-1].reshape(3,3)

  # Fの制約
  # 最後の特異値を0にして階数を2にする
  U,S,V = linalg.svd(F)
  S[2] = 0
  F = dot(U,dot(diag(S),V))

  return F

def compute_epipole(F):
  """ 基礎行列Fから（右側）のエピ極を計算する。
    （左のエピ極を計算するには F.Tを用いる """
  # Fの零空間(Fx=0)を返す
  U,S,V = linalg.svd(F)
  e = V[-1]
  return e/e[2]

def plot_epipolar_line(im,F,x,epipole=None,show_epipole=True):
  """ エピ極とエピポーラ線F*x=0を画像に描画する。
      Fは基礎行列、xは第2画像上の点 """

  m,n = im.shape[:2]
  line = dot(F,x)

  # エピポーラ線のパラメータと値
  t = linspace(0,n,100)
  lt = array([(line[2]+line[0]*tt)/(-line[1]) for tt in t])

  # 画像の中に含まれる線分だけを選ぶ
  ndx = (lt>=0) & (lt<m)
  plot(t[ndx],lt[ndx],linewidth=2)

  if show_epipole:
    if epipole is None:
      epipole = compute_epipole(F)
    plot(epipole[0]/epipole[2],epipole[1]/epipole[2],'r*')

def triangulate_point(x1,x2,P1,P2):
  """ 最小二乗解を用いて点の組を三角測量する """

  M = zeros((6,6))
  M[:3,:4] = P1
  M[3:,:4] = P2
  M[:3,4] = -x1
  M[3:,5] = -x2

  U,S,V = linalg.svd(M)
  X = V[-1,:4]

  return X / X[3]

def triangulate(x1,x2,P1,P2): 
  """ x1,x2(3*nの同次座標)の点の2視点三角測量 """

  n = x1.shape[1]
  if x2.shape[1] != n:
    raise ValueError("Number of points don't match.")

  X = [ triangulate_point(x1[:,i],x2[:,i],P1,P2) for i in range(n)]
  return array(X).T

def compute_P(x,X):
  """ 2D-3Dの対応の組(同次座標系)からカメラ行列を計算する """

  n = x.shape[1]
  if X.shape[1] != n:
    raise ValueError("Number of points don't match.")

  # DLT法で行列を作成する
  M = zeros((3*n,12+n))
  for i in range(n):
    M[3*i,0:4] = X[:,i]
    M[3*i+1,4:8] = X[:,i]
    M[3*i+2,8:12] = X[:,i]
    M[3*i:3*i+3,i+12] = -x[:,i]

  U,S,V = linalg.svd(M)

  return V[-1,:12].reshape((3,4))

def compute_P_from_fundamental(F):
  """ 第2のカメラ行列(P1 = [I 0] を仮定)を、
     基礎行列から計算する """

  e = compute_epipole(F.T) # left epipole
  Te = skew(e)
  return vstack((dot(Te,F.T).T,e)).T

def skew(a): 
  """ 任意のvについて a x v = Av になる交代行列A """
  return array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])

def compute_P_from_essential(E):
  """ 基本行列から第2のカメラ行列を計算する（P1＝[I 0]を仮定）
     出力は可能性のある4つのカメラ行列 """

  # Eの階数が2になるようにする
  U,S,V = svd(E)
  if det(dot(U,V))<0:
    V = -V
  E = dot(U,dot(diag([1,1,0]),V))

  # 行列を作成 (Hartley P.258参照)
  Z = skew([0,0,-1])
  W = array([[0,-1,0],[1,0,0],[0,0,1]])

  # 4つの解を返す
  P2 = [vstack((dot(U,dot(W,V)).T,U[:,2])).T,
        vstack((dot(U,dot(W,V)).T,-U[:,2])).T,
        vstack((dot(U,dot(W.T,V)).T,U[:,2])).T,
        vstack((dot(U,dot(W.T,V)).T,-U[:,2])).T]
  return P2

class RansacModel(object):
  """ ransac.py を用いて基礎行列を当てはめるためのクラス
    http://www.scipy.org/Cookbook/RANSAC """

  def __init__(self,debug=False):
    self.debug = debug

  def fit(self,data):
    """ 8つの選択した対応を使って基礎行列を推定する """

    # データを転置し2つの点群に分ける
    data = data.T
    x1 = data[:3,:8]
    x2 = data[3:,:8]

    # 基礎行列を推定して返す
    F = compute_fundamental_normalized(x1,x2)
    return F

  def get_error(self,data,F):
    """ すべての対応について x^T F x を計算し、
        変換された点の誤差を返す """

    # データを転置し2つの点群に分ける
    data = data.T
    x1 = data[:3]
    x2 = data[3:]

    # 誤差尺度としてSampson距離を使う
    Fx1 = dot(F,x1)
    Fx2 = dot(F,x2)
    denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
    err = ( diag(dot(x1.T,dot(F,x2))) )**2 / denom

    # 1点あたりの誤差を返す
    return err

def compute_fundamental_normalized(x1,x2):
  """ 正規化8点法を使って対応点群(x1,x2:3*nの配列)
      から基礎行列を計算する。各列は次のような並びである。
      [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

  n = x1.shape[1]
  if x2.shape[1] != n:
    raise ValueError("Number of points don't match.")

  # 画像の座標を正規化する
  x1 = x1 / x1[2]
  mean_1 = mean(x1[:2],axis=1)
  S1 = sqrt(2) / std(x1[:2])
  T1 = array([[S1,0,-S1*mean_1[0]],[0,S1,-S1*mean_1[1]],[0,0,1]])
  x1 = dot(T1,x1)

  x2 = x2 / x2[2]
  mean_2 = mean(x2[:2],axis=1)
  S2 = sqrt(2) / std(x2[:2])
  T2 = array([[S2,0,-S2*mean_2[0]],[0,S2,-S2*mean_2[1]],[0,0,1]])
  x2 = dot(T2,x2)

  # 正規化した座標でFを計算する
  F = compute_fundamental(x1,x2)

  # 正規化を元に戻す
  F = dot(T1.T,dot(F,T2))

  return F/F[2,2]


def F_from_ransac(x1,x2,model,maxiter=5000,match_theshold=1e-6):
  """ RANSAC(http://www.scipy.org/Cookbook/RANSAC のransac.py)
      を使って点の対応から基礎行列Fをロバスト推定する。
      入力：x1,x2(3*n配列) 同時座標系の点群 """

  import ransac

  data = vstack((x1,x2))

  # Fを計算しインライアのインデクスといっしょに返す
  F,ransac_data = ransac.ransac(data.T,model,8,maxiter,
                    match_theshold,20,return_all=True)
  return F, ransac_data['inliers']
