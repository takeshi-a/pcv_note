from pylab import *
from numpy import *
from scipy.ndimage import filters

def compute_harris_response(im,sigma=3):
   """ グレースケール画像の各ピクセルについて
       Harrisコーナー検出器の応答関数を定義する """

   # 微分係数
   imx = zeros(im.shape)
   filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
   imy = zeros(im.shape)
   filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)

   # Harris行列の成分を計算する
   Wxx = filters.gaussian_filter(imx*imx,sigma)
   Wxy = filters.gaussian_filter(imx*imy,sigma)
   Wyy = filters.gaussian_filter(imy*imy,sigma)

   # 判別式と対角成分
   Wdet = Wxx*Wyy - Wxy**2
   Wtr = Wxx + Wyy

   return Wdet / Wtr


def get_harris_points(harrisim,min_dist=10,threshold=0.1):
  """ Harris応答画像からコーナーを返す。
     min_distはコーナーや画像境界から分離する最小ピクセル数 """

  # 閾値thresholdを超えるコーナー候補を見つける
  corner_threshold = harrisim.max() * threshold
  harrisim_t = (harrisim > corner_threshold) * 1

  # 候補の座標を得る
  coords = array(harrisim_t.nonzero()).T

  # 候補の値を得る
  candidate_values = [harrisim[c[0],c[1]] for c in coords]

  # 候補をソートする
  index = argsort(candidate_values)

  # 許容する点の座標を配列に格納する
  allowed_locations = zeros(harrisim.shape)
  allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1

  # 最小距離を考慮しながら、最良の点を得る
  filtered_coords = []
  for i in index:
    if allowed_locations[coords[i,0],coords[i,1]] == 1:
      filtered_coords.append(coords[i])
      allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
          (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0

  return filtered_coords


def plot_harris_points(image,filtered_coords):
  """ 画像に見つかったコーナーを描画 """
  figure(figsize=(12,10))
  gray()
  imshow(image)
  plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
  axis('off')
  show()


def get_descriptors(image,filtered_coords,wid=5):
  """ 各点について、点の周辺で幅 2*wid+1 の近傍ピクセル値を返す。
      （点の最小距離 min_distance > wid を仮定する）"""
  desc = []
  for coords in filtered_coords:
    patch = image[coords[0]-wid:coords[0]+wid+1,
                  coords[1]-wid:coords[1]+wid+1].flatten() 
    desc.append(patch)

  return desc


def match(desc1,desc2,threshold=0.5):
  """ 正規化相互相関を用いて、第1の画像の各コーナー点記述子について、
      第2の画像の対応点を選択する。"""
  n = len(desc1[0])

  # 対応点ごとの距離
  d = -ones((len(desc1),len(desc2)))
  for i in range(len(desc1)):
    for j in range(len(desc2)):
      d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
      d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
      ncc_value = sum(d1 * d2) / (n-1)
      if ncc_value > threshold:
        d[i,j] = ncc_value

  ndx = argsort(-d)
  matchscores = ndx[:,0]

  return matchscores


def match_twosided(desc1,desc2,threshold=0.5):
  """ match()の双方向で一致を調べるバージョン """

  matches_12 = match(desc1,desc2,threshold)
  matches_21 = match(desc2,desc1,threshold)

  ndx_12 = where(matches_12 >= 0)[0]

  # 非対称の場合を除去する
  for n in ndx_12:
    if matches_21[matches_12[n]] != n:
      matches_12[n] = -1

  return matches_12


def appendimages(im1,im2):
  """ 2つの画像を左右に並べた画像を返す """

  # 行の少ない方を選び、空行を0で埋める
  rows1 = im1.shape[0]
  rows2 = im2.shape[0]

  if rows1 < rows2:
    im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
  elif rows1 > rows2:
    im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
  # 行が同じなら、0で埋める必要はない

  return concatenate((im1,im2), axis=1)


def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
  """ 対応点を線で結んで画像を表示する
    入力： im1,im2（配列形式の画像）、locs1,locs2（特徴点座標）
       machescores（match()の出力）、
       show_below（対応の下に画像を表示するならTrue）"""

  im3 = appendimages(im1,im2)
  if show_below:
    im3 = vstack((im3,im3))

  imshow(im3)

  cols1 = im1.shape[1]
  for i,m in enumerate(matchscores):
    if m>0: plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
  axis('off')

