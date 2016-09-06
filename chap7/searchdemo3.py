#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy import random
import cherrypy, os, urllib, pickle
import imagesearch

class SearchDemo(object):

  def __init__(self):
    # 画像リストを読み込む
    with open('webimlist.txt') as f:
      self.imlist = f.readlines()

    self.nbr_images = len(self.imlist)
    # self.ndx = range(self.nbr_images)
    self.ndx = list(range(self.nbr_images))

    # ボキャブラリを読み込む
    with open('vocabulary.pkl', 'rb') as f:
      self.voc = pickle.load(f)

    # 最大検索数を指定する
    self.maxres = 20

    # ヘッダーとフッターのHTML
    self.header = """
      <!doctype html>
      <html>
      <head>
      <title>Image search example</title>
      </head>
      <body>
      """
    self.footer = """
      </body>
      </html>
      """

  def index(self,query=None):
    self.src = imagesearch.Searcher('test.db',self.voc)

    html = self.header
    html += """
      <br />
      Click an image to search. <a href='?query='>Random selection</a> of images.
      <br /><br />
      """
    if query:
      # データベースに問い合わせ上位の画像を得る
      res = self.src.query(query)[:self.maxres]
      for dist,ndx in res:
        imname = self.src.get_filename(ndx)
        html += "<a href='?query="+imname+"'>"
        html += "<img src='"+imname+"' width='200' />"
        html += "</a>"
    else:
      # クエリがなければランダムに選択する
      random.shuffle(self.ndx)
      for i in self.ndx[:self.maxres]:
        imname = self.imlist[i]
        html += "<a href='?query="+imname+"'>"
        html += "<img src='"+imname+"' width='200' />"
        html += "</a>"

    html += self.footer
    return html

  index.exposed = True

if __name__ == '__main__':

  cherrypy.quickstart(SearchDemo(), '/',
      config=os.path.join(os.path.dirname(__file__), 'service.conf'))
