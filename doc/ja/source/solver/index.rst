順問題ソルバー
===============

順問題ソルバー ``Solver`` は探索パラメータ :math:`x` から最適化したい :math:`f(x)` を計算します。

現在 ODAT-SE に組み込まれているソルバーは ``analytical`` のみです。
2次元物質構造解析向けの順問題ソルバーは、別パッケージのラッパーとして提供されています（ :doc:`/start` を参照）。

``analytical`` ソルバー
~~~~~~~~~~~~~~~~~~~~~~~~

``analytical`` は探索アルゴリズムの性能評価を目的とした、
定義済みのベンチマーク関数 :math:`f(x)` を計算する ``Solver`` です。

入力パラメータ
^^^^^^^^^^^^^^^^^^^^^^^^

``solver`` セクション以下の ``function_name`` パラメータで用いる関数を指定します。

- ``function_name``

  形式: string型

  説明: 関数名。以下の関数が選べます。

  - ``quadratics``

    - 二次形式

      .. math::

          f(\vec{x}) = \sum_{i=1}^N x_i^2

    - 最適値は :math:`f(\vec{x}^*) = 0 \quad (\forall_i x_i^* = 0)`

  - ``rosenbrock``

    - `Rosenbrock 関数 <https://en.wikipedia.org/wiki/Rosenbrock_function>`_

    .. math::

      f(\vec{x}) = \sum_{i=1}^{N-1} \left[ 100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \right]

    - 最適値は :math:`f(\vec{x}^*) = 0 \quad (\forall_i x_i^* = 1)`

  - ``ackley``

    - `Ackley 関数 <https://en.wikipedia.org/wiki/Ackley_function>`_

    .. math::

      f(\vec{x}) = 20 + e - 20\exp\left[-0.2\sqrt{\frac{1}{N}\sum_{i=1}^N x_i^2}\right] - \exp\left[\frac{1}{N}\cos\left(2\pi x_i\right)\right]

    - 最適値は :math:`f(\vec{x}^*) = 0 \quad (\forall_i x_i^* = 0)`

  - ``himmelblau``

    - `Himmelblau 関数 <https://en.wikipedia.org/wiki/Himmelblau%27s_function>`_

    .. math::

      f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2

    - 最適値は :math:`f(3,2) = f(-2.805118, 3.131312) = f(-3.779310, -3.283186) = f(3.584428, -1.848126) = 0`
