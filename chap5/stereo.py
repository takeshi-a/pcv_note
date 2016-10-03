#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *
from scipy.ndimage import filters

def plane_sweep_ncc(im_l,im_r,start,steps,wid):
  """ 正規化相互相関を用いて視差画像を求める """

  m,n = im_l.shape

  # 差の和を格納する配列
  mean_l = zeros((m,n))
  mean_r = zeros((m,n))
  s = zeros((m,n))
  s_l = zeros((m,n))
  s_r = zeros((m,n))

  # 奥行き平面を格納する配列
  dmaps = zeros((m,n,steps))

  # パッチの平均を計算する
  filters.uniform_filter(im_l,wid,mean_l)
  filters.uniform_filter(im_r,wid,mean_r)

  # 画像を正規化する
  norm_l = im_l - mean_l
  norm_r = im_r - mean_r

  # 視差を順番に試していく
  for displ in range(steps):
    # 左の画像を右にずらして和を計算する
    filters.uniform_filter(roll(norm_l,-displ-start)*
                           norm_r,wid,s) # 分子の和
    filters.uniform_filter(roll(norm_l,-displ-start)*
                           roll(norm_l,-displ-start),wid,s_l)
    filters.uniform_filter(norm_r*norm_r,wid,s_r) # 分母の和

    # 相互相関の値を保存する
    dmaps[:,:,displ] = s/sqrt(s_l*s_r)

  # 各ピクセルで最良の奥行きを選ぶ
  return argmax(dmaps,axis=2)


def plane_sweep_gauss(im_l,im_r,start,steps,wid):
  """ ガウシアンで重み付けされた近傍画素の
      正規化相互相関を用いて視差画像を求める """

  m,n = im_l.shape

  # 差の和を格納する配列
  mean_l = zeros((m,n))
  mean_r = zeros((m,n))
  s = zeros((m,n))
  s_l = zeros((m,n))
  s_r = zeros((m,n))

  # 奥行き平面を格納する配列
  dmaps = zeros((m,n,steps))

  # パッチの平均を計算する
  filters.gaussian_filter(im_l,wid,0,mean_l)
  filters.gaussian_filter(im_r,wid,0,mean_r)

  # 画像を正規化する
  norm_l = im_l - mean_l
  norm_r = im_r - mean_r

  # 視差を順番に試していく
  for displ in range(steps):
    # 左の画像を右にずらして和を計算する
    filters.gaussian_filter(roll(norm_l,-displ-start)*
                            norm_r,wid,0,s) # 分子の和
    filters.gaussian_filter(roll(norm_l,-displ-start)*
                            roll(norm_l,-displ-start),wid,0,s_l)
    filters.gaussian_filter(norm_r*norm_r,wid,0,s_r) # 分母の和

    # 相互相関の値を保存する
    dmaps[:,:,displ] = s/sqrt(s_l*s_r)

  # 各ピクセルで最良の奥行きを選ぶ
  return argmax(dmaps,axis=2)
