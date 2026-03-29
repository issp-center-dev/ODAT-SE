開発者ガイド
=================

ODAT-SE は、順問題ソルバー ``Solver`` と探索アルゴリズム ``Algorithm`` を組み合わせて逆問題を解くフレームワークです。
定義済みの ``Solver`` や ``Algorithm`` に加えて、ユーザーが独自のものを定義して利用できます。

.. code-block:: text

   Algorithm ----> Runner ----> Solver
   (search)        (bridge)      (objective function)
       |              |              |
       |  propose x   | transform    | compute f(x)
       |              | & check      | and return
       |<-------------|<-------------|

本章では、最も一般的なユースケースである **独自のソルバーを追加する方法** をチュートリアル形式で解説した後、
APIの詳細を説明します。

:doc:`tutorial_solver`
    独自の目的関数を ODAT-SE で最適化するチュートリアルです。コピペして動かせる完全な例を用いて、ファイル作成から実行・結果確認までの手順を説明します。プログラムに不慣れな方はまずこちらをお読みください。

:doc:`solver`
    ``Solver`` クラスの API リファレンスです。``SolverBase`` を継承して ``evaluate`` メソッドを実装する方法を詳しく解説します。

:doc:`algorithm`
    ``Algorithm`` クラスの API リファレンスです。独自の探索アルゴリズムを定義する方法を解説します。

:doc:`common`
    ``Info``, ``Runner``, ``Mapping``, ``Limitation`` など、Solver と Algorithm に共通するクラスの説明です。

:doc:`usage`
    カスタム Solver / Algorithm を組み合わせて実行する際のコード例です。

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorial_solver
   solver
   algorithm
   common
   usage
