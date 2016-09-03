#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image
from pylab import *

from xml.dom import minidom

def read_points_from_xml(xmlFileName):
  """ 顔の位置合わせ用の制御点を読み込む """

  xmldoc = minidom.parse(xmlFileName)
  facelist = xmldoc.getElementsByTagName('face')
  faces = {}
  for xmlFace in facelist:
    fileName = xmlFace.attributes['file'].value
    xf = int(xmlFace.attributes['xf'].value)
    yf = int(xmlFace.attributes['yf'].value)
    xs = int(xmlFace.attributes['xs'].value)
    ys = int(xmlFace.attributes['ys'].value)
    xm = int(xmlFace.attributes['xm'].value)
    ym = int(xmlFace.attributes['ym'].value)
    faces[fileName] = array([xf, yf, xs, ys, xm, ym])

#    im = array(Image.open(os.path.join(os.path.dirname(xmlFileName),fileName)))
#    imshow(im)
#    plot([xf,xs,xm],[yf,ys,ym],'*')
#    title(fileName)
#    show()

  return faces

from scipy import linalg

def compute_rigid_transform(refpoints,points):
  """ pointsをrefpointsに対応づける回転と拡大率、
      平行移動を計算する """

  A = array([ [points[0], -points[1], 1, 0],
              [points[1], points[0], 0, 1],
              [points[2], -points[3], 1, 0],
              [points[3], points[2], 0, 1],
              [points[4], -points[5], 1, 0],
              [points[5], points[4], 0, 1]])

  y = array([ refpoints[0],
              refpoints[1],
              refpoints[2],
              refpoints[3],
              refpoints[4],
              refpoints[5]])

  # ||Ax - y||を最小化する最小二乗解
  a,b,tx,ty = linalg.lstsq(A,y)[0]
  R = array([[a, -b], [b, a]]) # 回転行列に拡大率も含まれている

  return R,tx,ty

from scipy import ndimage
from scipy.misc import imsave
import os

def rigid_alignment(faces,path,plotflag=False):
  """ 画像を位置合わせし、新たな画像として保存する。
      pathは、位置合わせした画像の保存先
      plotflag=Trueなら、画像を表示する """

  # 最初の画像の点を参照点とする
  # refpoints = faces.values()[0]
  refpoints = list(faces.values())[0]

  # 各画像を相似変換で変形する
  for face in faces:
    points = faces[face]

    R,tx,ty = compute_rigid_transform(refpoints, points)
    T = array([[R[1][1], R[1][0]], [R[0][1], R[0][0]]])

    im = array(Image.open(os.path.join(path,face)))
    im2 = zeros(im.shape, 'uint8')

    # 色チャンネルごとに変形する
    for i in range(len(im.shape)):
      im2[:,:,i] = ndimage.affine_transform(im[:,:,i],linalg.inv(T),
                                            offset=[-ty,-tx])
    if plotflag:
      imshow(im2)
      show()

    # 境界で切り抜き、位置合わせした画像を保存する
    h,w = im2.shape[:2]
    border = (w+h)/20
    imsave(os.path.join(path, 'aligned/'+face),
          im2[border:h-border,border:w-border,:])
