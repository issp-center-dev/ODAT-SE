.. post-tools documentation master file, created by
   sphinx-quickstart on Wed Mar  5 21:21:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ポスト処理ツール
========================

PAMC計算のポスト処理ツールと解析の流れを解説する。

以下のツールが script ディレクトリに用意されている。
次節では例を用いて解析の流れを紹介する。
個別のツールについて詳細はリファレンスの各項目を参照のこと。

**extract_combined.py**
    combined 形式で出力されたログファイルから特定の項目を取り出す。

**plt_1D_histogram.py**
    1次元周辺化ヒストグラムを作成する。

**plt_2D_histogram.py**
    2次元周辺化ヒストグラムを作成する。

**plt_model_evidence.py**
    model evidence を計算する。

**separateT.py**
    MCMCのログファイルを温度点ごとに分割する。

**summarize_each_T.py**
    MCMCが出力するログファイルからannealing後のレプリカの情報を取り出し、温度点ごとに集約する。

.. toctree::
   :maxdepth: 2

   tutorial
   tools/index
