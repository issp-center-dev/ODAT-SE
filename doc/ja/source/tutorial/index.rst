チュートリアル
==================================

このチュートリアルでは、解析関数の最小化問題を例として、ODAT-SEによる逆問題解析の方法を説明します。
ODAT-SEには、逆問題を解くためのアルゴリズムとして以下の6つの手法が用意されています。

- ``minsearch``

  Nelder-Mead法

- ``mapper_mpi``

  与えられたパラメータの探索グリッドを全探索する

- ``random_search``

  与えられた範囲からランダムにパラメータを選び、関数値を評価する

- ``bayes``

  ベイズ最適化

- ``ttopt``

  テンソル列最適化

- ``exchange``

  レプリカ交換モンテカルロ法

- ``pamc``

  ポピュレーションアニーリング法

以下ではこれらのアルゴリズムを用いた実行方法を説明します。
また、制約式を用いて探索範囲を制限できる ``[runner.limitation]`` セクションを使用した実行方法も説明しています。
最後に、自分で順問題ソルバーを定義する簡単な例について説明します。

実行前の準備
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

チュートリアルの実行には、ODAT-SE のインストール (:doc:`../start` 参照) に加えてサンプルファイルが必要です。
サンプルファイルはソースパッケージに同梱されています。
以下のようにリポジトリを取得し、取得したディレクトリに移動してください。

.. code-block::

    $ git clone https://github.com/issp-center-dev/ODAT-SE.git
    $ cd ODAT-SE

各チュートリアルの手順は、この ODAT-SE ディレクトリの直下から始めることを想定しています。

.. toctree::
  :maxdepth: 1

  intro
  minsearch
  mapper
  random_search
  bayes
  ttopt
  exchange
  pamc
  limitation
  solver_simple
  parallel_solver
