#!/usr/bin/python
# -*- coding: utf-8 -*-

from PIL import Image
from pylab import *
import pickle
from sqlite3 import dbapi2 as sqlite

class Indexer(object):

  def __init__(self,db,voc):
    """ データベース名とボキャブラリオブジェクトを
       用いて初期化する """

    self.con = sqlite.connect(db)
    self.voc = voc

  def __del__(self):
    self.con.close()

  def db_commit(self):
    self.con.commit()

  def create_tables(self):
    """ データベースのテーブルを作成する """
    self.con.execute('create table imlist(filename)')
    self.con.execute('create table imwords(imid,wordid,vocname)')
    self.con.execute('create table imhistograms(imid,histogram,vocname)')
    self.con.execute('create index im_idx on imlist(filename)')
    self.con.execute('create index wordid_idx on imwords(wordid)')
    self.con.execute('create index imid_idx on imwords(imid)')
    self.con.execute('create index imidhist_idx on imhistograms(imid)')
    self.db_commit()


  def add_to_index(self,imname,descr):
    """ 画像と特徴量記述子を入力し、ボキャブラリに
       射影して、データベースに追加する """

    if self.is_indexed(imname): return
    print('indexing', imname)

    # 画像IDを取得する
    imid = self.get_id(imname)

    # ワードを取得する
    imwords = self.voc.project(descr)
    nbr_words = imwords.shape[0]

    # 各ワードを画像に関係づける
    for i in range(nbr_words):
      word = imwords[i]
      # ワード番号をワードIDとする
      self.con.execute("insert into imwords(imid,wordid,vocname) values (?,?,?)", (imid,word,self.voc.name))

    # 画像のワードヒストグラムを記録する
    # NumPyの配列を文字列に変換するためにpickleを用いる
    self.con.execute("insert into imhistograms(imid,histogram,vocname) values (?,?,?)", (imid,pickle.dumps(imwords),self.voc.name))

  def is_indexed(self,imname):
    """ imnameがインデクスを持っていればTrueを返す """

    im = self.con.execute("select rowid from imlist where filename='%s'" % imname).fetchone()
    return im != None

  def get_id(self,imname):
    """ 成分のIDを取得する。なければ追加する """

    cur = self.con.execute(
      "select rowid from imlist where filename='%s'" % imname)
    res=cur.fetchone()
    if res==None:
      cur = self.con.execute(
        "insert into imlist(filename) values ('%s')" % imname)
      return cur.lastrowid
    else:
      return res[0]


class Searcher(object):

  def __init__(self,db,voc):
    """ データベース名とボキャブラリを用いて初期化する """
    self.con = sqlite.connect(db)
    self.voc = voc

  def __del__(self):
    self.con.close()

  def candidates_from_word(self,imword):
    """ imwordを含む画像のリストを取得する """

    im_ids = self.con.execute(
      "select distinct imid from imwords where wordid=%d" % imword).fetchall()
    return [i[0] for i in im_ids]

  def candidates_from_histogram(self,imwords):
    """ 複数の類似ワードを持つ画像のリストを取得する """

    # ワードのIDを取得する
    words = imwords.nonzero()[0]

    # 候補を見つける
    candidates = []
    for word in words:
      c = self.candidates_from_word(word)
      candidates+=c

    # 全ワードを重複なく抽出し、出現回数の大きい順にソートする
    tmp = [(w,candidates.count(w)) for w in set(candidates)]
    # Python 3.xではcmpは使えないため、置き換える
    # tmp.sort(cmp=lambda x,y:cmp(x[1],y[1]))
    tmp.sort(key=lambda x: x[1])
    tmp.reverse()

    # 一致するものほど先になるようにソートしたリストを返す
    return [w[0] for w in tmp]

  def get_imhistogram(self,imname):
    """ 画像のワードヒストグラムを返す """
    im_id = self.con.execute(
      "select rowid from imlist where filename='%s'" % imname).fetchone()
    s = self.con.execute(
      "select histogram from imhistograms where rowid='%d'" % im_id).fetchone()

    # pickleを使って文字列をNumPy配列に変換する
    # python 3.xではpickle.loads()に文字列ではなく、バイナリを入力する
    # return pickle.loads(str(s[0]))
    return pickle.loads((s[0]))
    
  def query(self,imname):
    """ imnameの画像に一致する画像のリストを見つける """

    h = self.get_imhistogram(imname)
    candidates = self.candidates_from_histogram(h)

    matchscores = []
    for imid in candidates:
      # 名前を取得する
      cand_name = self.con.execute(
        "select filename from imlist where rowid=%d" % imid).fetchone()
      cand_h = self.get_imhistogram(cand_name)
      cand_dist = sqrt( sum( (h-cand_h)**2 ) ) # L2距離を用いる
      matchscores.append( (cand_dist,imid) )

    # 距離の小さい順にソートした距離と画像IDを返す
    matchscores.sort()
    return matchscores

  def get_filename(self,imid):
    """ 画像IDに対応するファイル名を返す """
  
    s = self.con.execute(
      "select filename from imlist where rowid='%d'" % imid).fetchone()
    return s[0]

def compute_ukbench_score(src,imlist):
  """ 検索結果の上位4件のうち、正解数の平均を返す """

  nbr_images = len(imlist)
  pos = zeros((nbr_images,4))
  # 各画像の検索結果の上位4件を得る
  for i in range(nbr_images):
    pos[i] = [w[1]-1 for w in src.query(imlist[i])[:4]]

  # スコアを計算して平均を返す
  score = array([ (pos[i]//4)==(i//4) for i in range(nbr_images)])*1.0
  return sum(score) / (nbr_images)

def plot_results(src,res):
  """ 検索結果リスト'res'の画像を表示する """

  figure(figsize=(15,15))
  nbr_results = len(res)
  for i in range(nbr_results):
    imname = src.get_filename(res[i])
    subplot(1,nbr_results,i+1)
    imshow(array(Image.open(imname)))
    axis('off')
  show()

