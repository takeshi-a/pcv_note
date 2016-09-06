#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *
from scipy.cluster.vq import *
import sift

class Vocabulary(object):

  def __init__(self,name):
    self.name = name
    self.voc = []
    self.idf = []
    self.trainingdata = []
    self.nbr_words = 0

  def train(self,featurefiles,k=100,subsampling=10):
    """ featurefilesに列挙されたファイルから特徴量を読み込み
      k平均法とk個のビジュアルワードを用いてボキャブラリを
      学習する。subsamplingで教師データを間引いて高速化可能 """

    nbr_images = len(featurefiles)
    # ファイルから特徴量を読み込む
    descr = []
    descr.append(sift.read_features_from_file(featurefiles[0])[1])
    descriptors = descr[0] #stack all features for k-means
    for i in arange(1,nbr_images):
      descr.append(sift.read_features_from_file(featurefiles[i])[1])
      descriptors = vstack((descriptors,descr[i]))

    # k平均法：最後の数字で試行数を指定する
    self.voc,distortion = kmeans(descriptors[::subsampling,:],k,1)
    self.nbr_words = self.voc.shape[0]

    # 教師画像を順番にボキャブラリに射影する
    imwords = zeros((nbr_images,self.nbr_words))
    for i in range( nbr_images ):
      imwords[i] = self.project(descr[i])

    nbr_occurences = sum( (imwords > 0)*1 ,axis=0)

    self.idf = log( (1.0*nbr_images) / (1.0*nbr_occurences+1) )
    self.trainingdata = featurefiles

  def project(self,descriptors):
    """ 記述子をボキャブラリに射影して、
        単語のヒストグラムを作成する """

    # ビジュアルワードのヒストグラム
    imhist = zeros((self.nbr_words))
    words,distance = vq(descriptors,self.voc)
    for w in words:
      imhist[w] += 1

    return imhist
