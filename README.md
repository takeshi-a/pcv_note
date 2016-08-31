# 『実践コンピュータビジョン』演習ノート

## 目的
Jan Erik Solem著、『実践コンピュータビジョン（以下、PCV）』のサンプルコードをIPython notebookでトレースして、コンピュータビジョンの基本を学ぶ。

## テキスト
Jan Erik Solem（著）, 相川愛三（訳）, 『実践コンピュータビジョン』, O'Reilly Japan (2012)  
[http://www.oreilly.co.jp/books/9784873116075/](http://www.oreilly.co.jp/books/9784873116075/)

## サポートページ
『実践コンピュータビジョン サンプルプログラム』<br>
訳者によるサンプルプログラムの説明が掲載された公式サポートページ。<br>
[http://www.oreilly.co.jp/pub/9784873116075/](http://www.oreilly.co.jp/pub/9784873116075/)  

## 著者のGitHubサイト
Jan Erik Solem氏のGitHubアカウントのサイトに、PCVテキストのサンプルコードが掲載されている。<br>
[https://github.com/jesolem/PCV](https://github.com/jesolem/PCV)

## importのルール 
PCVではnumpy, pylabなどのパッケージをimportする際、\*（アスタリスク）を用いた一括importが多用されている。  
この演習ノートでは、できるだけ利用しているパッケージを明確にするため、\*を用いない。

```python
# PCVテキスト内の一般的なimport

from PIL import Image
from pylab import *
from numpy import *

# 演習ノート内の通常のimport方法
from PIL import Image
import numpy as numpy
import matplotlib.pyplot as plt
```

## Pythonのバージョン
PCVテキストではPython2.xを基本として、プログラムが書かれているが、  
本ノートでは、Python 3.xを基本として、コーディングを行う。  
Python 3.xに対応するには、print文、文字列の扱いなど、いくつかの変更が必要だが、ノート内でできる限り説明を加える。

## 画像ファイルのダウンロード
このレポジトリには、サンプルコードの実行に必要なファイルの一部のみを掲載している。
大量の画像を保存することができないので、基本的にデータセットはそれぞれのnotebookに記載されたオリジナルの提供先を参照されたい。

## PCVのオリジナルモジュール
PCVテキスト内では画像処理のために、独自のモジュールが用意されている。各章で必要なモジュールのスクリプトファイルを配置している。ファイルは基本的にサポートページから引用しているが、Python 3.xで動作するように一部のコードを改編している。

## 謝辞
『実践コンピュータビジョン』という素晴らしい本を執筆されたJan Erik Solem氏に感謝します。<br>
またこの著書の内容を丁寧に翻訳して、日本語の完璧なサンプルコードを提供している相川愛三氏、本著の出版元でO'Reilly Japanのみなさまに感謝します。
この演習ノートが少しでもPCVを学ぶ人のお役に立てば、幸いです。

