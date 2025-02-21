.. 2dmat documentation master file, created by
   sphinx-quickstart on Tue May 26 18:44:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

チュートリアル
==================================

このチュートリアルでは、解析関数の最小化問題を例として、ODAT-SEによる逆問題解析の方法を説明します。
ODAT-SEには、逆問題を解くためのアルゴリズムとして以下の5つの手法が用意されています。

- ``minsearch``

  Nealder-Mead法

- ``mapper_mpi``

  与えられたパラメータの探索グリッドを全探索する

- ``bayes``

  ベイズ最適化

- ``exchange``

  レプリカ交換モンテカルロ法

- ``pamc``

  ポピュレーションアニーリング法

以下ではこれらのアルゴリズムを用いた実行方法を説明します。
また、制約式を用いて探索範囲を制限できる ``[runner.limitation]`` セクションを使用した実行方法も説明しています。
最後に、自分で順問題ソルバーを定義する簡単な例について説明します。

.. toctree::
   :maxdepth: 1

   intro
   minsearch
   mapper
   bayes
   exchange
   pamc
   limitation
   solver_simple
