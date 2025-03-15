.. post-tools documentation master file, created by
   sphinx-quickstart on Wed Mar  5 21:21:56 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ポスト処理ツール
========================

この章ではPAMC計算のポスト処理ツールと解析の流れを解説します。

以下のツールが script ディレクトリに用意されています。
チュートリアルでは例を用いて解析の流れを紹介します。。
個別のツールについての詳細は、リファレンスの各項目を参照してください。

:doc:`tools/extract_combined`
    combined 形式で出力されたログファイルから特定の項目を取り出します。

:doc:`tools/plt_1D_histogram`
    1次元周辺化ヒストグラムを作成します。

:doc:`tools/plt_2D_histogram`
    2次元周辺化ヒストグラムを作成します。

:doc:`tools/plt_model_evidence`
    model evidence を計算します。

:doc:`tools/separateT`
    MCMCのログファイルを温度点ごとに分割します。

:doc:`tools/summarize_each_T`
    MCMCが出力するログファイルからannealing後のレプリカの情報を取り出し、温度点ごとに集約します。

.. toctree::
   :maxdepth: 2

   tutorial
   tools/index
