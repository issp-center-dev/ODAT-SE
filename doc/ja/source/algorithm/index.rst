.. 2dmat documentation master file, created by
   sphinx-quickstart on Tue May 26 18:44:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

探索アルゴリズム
====================

探索アルゴリズム ``Algorithm`` は ``Solver`` の結果 :math:`f(x)` を用いて
パラメータ空間 :math:`\mathbf{X} \ni x` を探索し、 :math:`f(x)` の最小化問題を解きます。
ODAT-SE では以下の探索アルゴリズムが利用できます。
各項目をクリックすると、入力パラメータや出力ファイルなどの詳細な使用方法を確認できます。

:doc:`minsearch`
    Nelder-Mead法（シンプレックス法）による最適化を行います。勾配を使わない直接探索法で、少数のパラメータに対して高速に収束します。scipy を利用します。

:doc:`mapper_mpi`
    パラメータ空間をグリッド状に分割し、すべての格子点で :math:`f(x)` を評価します。MPI による並列化に対応しており、パラメータ空間の全体像を把握するのに適しています。

:doc:`exchange`
    レプリカ交換モンテカルロ法（パラレルテンパリング）により探索を行います。異なる温度のレプリカ間で配置を交換することで、局所解への捕捉を回避します。mpi4py を利用します。

:doc:`pamc`
    ポピュレーションアニーリングモンテカルロ法により探索を行います。多数のレプリカ（ウォーカー）を徐々に冷却しながらリサンプリングすることで、効率的にパラメータ空間を探索します。

:doc:`bayes`
    ベイズ最適化により探索を行います。ガウス過程回帰を用いて :math:`f(x)` のサロゲートモデルを構築し、獲得関数に基づいて次の評価点を選択します。少数の評価回数で効率的に最適解を探索できます。physbo を利用します。

:doc:`random_search`
    ランダムにパラメータを選んで目的関数を評価します。指定した範囲から一様乱数によりパラメータをサンプリングし、パラメータ空間の大域的な概観を把握するのに適しています。MPI による並列化に対応しています。

.. toctree::
   :maxdepth: 1
   :hidden:

   minsearch
   mapper_mpi
   random_search
   exchange
   pamc
   bayes
   random_search
