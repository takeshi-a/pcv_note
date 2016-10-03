#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt
from itertools import combinations

class ClusterNode(object):
  def __init__(self,vec,left,right,distance=0.0,count=1):
    self.left = left
    self.right = right
    self.vec = vec
    self.distance = distance
    self.count = count # 重み付き平均に使う

  def extract_clusters(self,dist):
    """ 階層クラスタリングツリーのdistance<distの
        クラスタ部分ツリーをリストに展開する """
    if self.distance < dist:
      return [self]
    return self.left.extract_clusters(dist) + \
           self.right.extract_clusters(dist)

  def get_cluster_elements(self):
    """ クラスタの部分ツリーの要素のIDを返す """
    return self.left.get_cluster_elements() + \
           self.right.get_cluster_elements()

  def get_height(self):
    """ ノードの高さ（枝の和）を返す """
    return self.left.get_height() + self.right.get_height()

  def get_depth(self):
    """ ノードの深さ（子ノードの最大深さ＋自分の距離）"""
    return max(self.left.get_depth(), self.right.get_depth()) + \
               self.distance

  def draw(self,draw,x,y,s,imlist,im):
    """ 再帰的にノードを描画する """

    h1 = int(self.left.get_height()*20 / 2)
    h2 = int(self.right.get_height()*20 /2)
    top = y-(h1+h2)
    bottom = y+(h1+h2)

    # 子ノード間の垂直線
    draw.line((x,top+h1,x,bottom-h2),fill=(0,0,0))

    # 子ノードへの水平線
    ll = self.distance*s
    draw.line((x,top+h1,x+ll,top+h1),fill=(0,0,0))
    draw.line((x,bottom-h2,x+ll,bottom-h2),fill=(0,0,0))

    # 左右の子ノードを再帰的に描画する
    self.left.draw(draw,x+ll,top+h1,s,imlist,im)
    self.right.draw(draw,x+ll,bottom-h2,s,imlist,im)

class ClusterLeafNode(object):
  def __init__(self,vec,id):
    self.vec = vec
    self.id = id

  def extract_clusters(self,dist):
    return [self]

  def get_cluster_elements(self):
    return [self.id]

  def get_height(self):
    return 1

  def get_depth(self):
    return 0

  def draw(self,draw,x,y,s,imlist,im):
    """ 葉ノードにサムネイル画像を表示する """
    nodeim = Image.open(imlist[self.id])
    nodeim.thumbnail([20,20])
    ns = nodeim.size
    im.paste(nodeim,[int(x),int(y-ns[1]//2),
                     int(x+ns[0]),int(y+ns[1]-ns[1]//2)])

def L2dist(v1,v2):
  return sqrt(sum((v1-v2)**2))

def L1dist(v1,v2):
  return sum(abs(v1-v2))

def hcluster(features,distfcn=L2dist):
  """ 特徴量の並びを階層クラスタリングする """

  # 距離計算のキャッシュ
  distances = {}

  # 各要素をクラスタとして初期化する
  node = [ClusterLeafNode(array(f),id=i)
          for i,f in enumerate(features)]

  while len(node)>1:
    closest = float('Inf')

    # すべての組を調べ、最小距離を求める
    for ni,nj in combinations(node,2):
      if (ni,nj) not in distances:
        distances[ni,nj] = distfcn(ni.vec,nj.vec)

      d = distances[ni,nj]
      if d<closest:
        closest = d
        lowestpair = (ni,nj)
    ni,nj = lowestpair

    # 2つのクラスタの平均をとる
    new_vec = (ni.vec + nj.vec) / 2.0

    # 新しいノードを生成する
    new_node = ClusterNode(new_vec,left=ni,right=nj,
                           distance=closest)
    node.remove(ni)
    node.remove(nj)
    node.append(new_node)

  return node[0]

from PIL import Image,ImageDraw

def draw_dendrogram(node,imlist,filename='clusters.jpg'):
  """ クラスタの系統図を描画してファイルに保存する """

  # 高さと幅
  rows = node.get_height()*20
  cols = 1200

  # 画像の幅に合わせるための縮尺
  s = float(cols-150)/node.get_depth()

  # 画像と描画オブジェクトを生成する
  im = Image.new('RGB',(cols,rows),(255,255,255))
  draw = ImageDraw.Draw(im)

  # ツリーの起点の最初の線
  draw.line((0,rows/2,20,rows/2),fill=(0,0,0))

  # ノードを再帰的に描画する
  node.draw(draw,20,(rows/2),s,imlist,im)
  im.save(filename)
  plt.figure(figsize=(10,10))
  im = array(im)
  plt.imshow(im)
  plt.axis('off')
  plt.show()
  # im.show()


