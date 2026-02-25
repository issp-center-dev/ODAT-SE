.. 2dmat documentation master file, created by
   sphinx-quickstart on Tue May 26 18:44:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

(開発者向け)ユーザー定義アルゴリズム・ソルバー
=================================================

本ソフトウェアは、順問題のためのソルバー ``Solver`` と最適化のためのアルゴリズム ``Algorithm`` を組み合わせることで全体の逆問題を解きます。
``Solver`` と ``Algorithm`` はすでに定義済みのものがいくつかありますが、これらを自分で定義することで、適用範囲を広げられます。
本章では、 ``Solver`` や ``Algorithm`` を定義する方法およびこれらを使用する方法を解説します。


.. toctree::
   :maxdepth: 1

   common
   solver
   algorithm
   usage
