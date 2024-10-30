.. 2dmat documentation master file, created by
   sphinx-quickstart on Tue May 26 18:44:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

チュートリアル
==================================

順問題ソルバーとして用意されている ``sim_trhepd_rheed`` は東北大学の花田貴先生によって開発された
物質反射高速（陽）電子回折(RHEED, TRHEPD)の解析ソフトウェアをベースに作成されています。
TRHEPDでは原子座標を与えた場合に、回折データがシミュレーション結果として与えられます。
そのため、原子座標から回折データへの順問題を取り扱っているといえます。
一方、多くの場合回折データは実験で与えられ、それを再現するような原子座標などが求められます。
これらは上記の順問題に対して、逆問題に相当します。

本ソフトウェアでは逆問題を解くためのアルゴリズムとして

- ``minsearch``

  Nealder-Mead法を用いもっともらしい原子座標を推定

- ``mapper_mpi``

  与えられたパラメータの探索グリッドを全探索することで、もっともらしい原子座標を推定

- ``bayes``

  ベイズ最適化を用いもっともらしい原子座標を推定

- ``exchange``

  レプリカ交換モンテカルロ法を用いてもっともらしい原子座標をサンプリング

- ``pamc``

  ポピュレーションアニーリング法を用いてもっともらしい原子座標をサンプリング

の5つのアルゴリズムが用意されています。
本チュートリアルでは、最初に順問題プログラム ``sim_trhepd_rheed`` の実行方法、
その後に ``minsearch`` , ``mapper_mpi``, ``bayes``, ``exchange``, ``pamc`` の実行方法について順に説明します。
また、制約式を用いて探索範囲を制限出来る ``[runner.limitation]`` セクションを使用した実行方法も説明しています。
最後に、自分で順問題ソルバーを定義する簡単な例について説明します。

.. toctree::
   :maxdepth: 1

   sim_trhepd_rheed
   minsearch
   mpi
   bayes
   exchange
   limitation
   pamc
   solver_simple
   
