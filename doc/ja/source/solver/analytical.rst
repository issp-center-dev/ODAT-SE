``analytical`` ソルバー
************************

``analytical`` は探索アルゴリズムの性能評価を目的とした、
定義済みのベンチマーク関数 :math:`f(x)` を計算する ``Solver`` です。

入力パラメータ
~~~~~~~~~~~~~~~~~~~~~~~~

``solver`` セクション以下の ``function_name`` パラメータで用いる関数を指定します。

- ``function_name``

  形式: string型

  説明: 関数名。以下の関数が選べます。

  - ``quadratics``

    - 二次形式（球面関数）

      .. math::

          f(\vec{x}) = \sum_{i=1}^N x_i^2

    - 大域最適値: :math:`f(\vec{x}^*) = 0 \quad (\forall_i\, x_i^* = 0)`

  - ``quartics``

    - 2つの大域最適解をもつ4次関数

      .. math::

          f(\vec{x}) = \left(\frac{1}{N}\sum_{i=1}^N (x_i - 1)^2\right) \left(\frac{1}{N}\sum_{i=1}^N (x_i + 1)^2\right)

    - 大域最適値: :math:`f(\vec{x}^*) = 0 \quad (\forall_i\, x_i^* = 1` および :math:`\forall_i\, x_i^* = -1)`
    - 鞍点: :math:`f(\vec{0}) = 1`

  - ``ackley``

    - `Ackley 関数 <https://en.wikipedia.org/wiki/Ackley_function>`_

      .. math::

        f(\vec{x}) = 20 + e - 20\exp\!\left[-0.2\sqrt{\frac{1}{N}\sum_{i=1}^N x_i^2}\right] - \exp\!\left[\frac{1}{N}\sum_{i=1}^N\cos(2\pi x_i)\right]

    - 大域最適値: :math:`f(\vec{x}^*) = 0 \quad (\forall_i\, x_i^* = 0)` 。多数の局所最適解を持ちます。

  - ``alpine``

    - Alpine 関数

      .. math::

          f(\vec{x}) = \sum_{i=1}^N \left|x_i \sin(x_i) + 0.1\, x_i\right|

    - 大域最適値: :math:`f(\vec{x}^*) = 0 \quad (\forall_i\, x_i^* = 0)`

  - ``exponential``

    - 指数関数

      .. math::

          f(\vec{x}) = -\exp\!\left(-\frac{1}{2}\sum_{i=1}^N x_i^2\right)

    - 大域最適値: :math:`f(\vec{x}^*) = -1 \quad (\forall_i\, x_i^* = 0)`

  - ``griewank``

    - Griewank 型関数

      .. math::

          f(\vec{x}) = 1 + \frac{1}{4000}\sum_{i=1}^N x_i^2 + \prod_{i=1}^N \cos\!\left(\frac{x_i}{\sqrt{i}}\right)

    - 注意: 余弦積の項を加算（減算ではない）しているため、標準の Griewank 関数とは異なります。

  - ``himmelblau``

    - `Himmelblau 関数 <https://en.wikipedia.org/wiki/Himmelblau%27s_function>`_ （:math:`N = 2` のみ）

      .. math::

          f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    - 大域最適値: :math:`f = 0` の点が4つあります: :math:`(3,\,2)`, :math:`(-2.805118,\,3.131312)`, :math:`(-3.779310,\,-3.283186)`, :math:`(3.584428,\,-1.848126)`

  - ``michalewicz``

    - `Michalewicz 関数 <https://en.wikipedia.org/wiki/Michalewicz_function>`_

      .. math::

          f(\vec{x}) = -\sum_{i=1}^N \sin(x_i)\left[\sin\!\left(\frac{i\, x_i^2}{\pi}\right)\right]^{20}

    - 大域最適値および位置は次元に依存します。局所最適解は :math:`d!` 個存在します。
      :math:`d = 2` のとき :math:`f(\vec{x}^*) \approx -1.8013 \quad (\vec{x}^* \approx (2.2051,\,1.5698))`。
      :math:`d = 5` のとき :math:`f(\vec{x}^*) \approx -4.6876`。
      :math:`d = 10` のとき :math:`f(\vec{x}^*) \approx -9.6602`。

  - ``qing``

    - Qing 関数

      .. math::

          f(\vec{x}) = \sum_{i=1}^N \left(x_i^2 - i\right)^2

    - 大域最適値: :math:`f(\vec{x}^*) = 0 \quad (x_i^* = \pm\sqrt{i},\; i = 1, \ldots, N)`

  - ``rastrigin``

    - `Rastrigin 関数 <https://en.wikipedia.org/wiki/Rastrigin_function>`_

      .. math::

          f(\vec{x}) = 10N + \sum_{i=1}^N \left[x_i^2 - 10\cos(2\pi x_i)\right]

    - 大域最適値: :math:`f(\vec{x}^*) = 0 \quad (\forall_i\, x_i^* = 0)`

  - ``rosenbrock``

    - `Rosenbrock 関数 <https://en.wikipedia.org/wiki/Rosenbrock_function>`_

      .. math::

          f(\vec{x}) = \sum_{i=1}^{N-1} \left[100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2\right]

    - 大域最適値: :math:`f(\vec{x}^*) = 0 \quad (\forall_i\, x_i^* = 1)`

  - ``schaffer``

    - Schaffer 関数（一般化版）

      .. math::

          f(\vec{x}) = \sum_{i=1}^{N-1} \left[0.5 + \frac{\sin^2\!\left(x_i^2 + x_{i+1}^2\right) - 0.5}{\left(1 + 0.001\,(x_i^2 + x_{i+1}^2)\right)^2}\right]

    - 大域最適値: :math:`f(\vec{x}^*) = 0 \quad (\forall_i\, x_i^* = 0)`

  - ``schwefel``

    - `Schwefel 関数 <https://en.wikipedia.org/wiki/Schwefel_function>`_

      .. math::

          f(\vec{x}) = 418.9829\,N - \sum_{i=1}^N x_i \sin\!\left(\sqrt{|x_i|}\right)

    - 大域最適値: :math:`f(\vec{x}^*) \approx 0 \quad (\forall_i\, x_i^* \approx 420.9687)`

  - ``linear_regression_test``

    - ガウスノイズ :math:`\mathcal{N}(0, \sigma^2)` をもつ線形回帰モデル :math:`y = at + b` の負の対数尤度。
      データ :math:`\{(t_k, y_k)\} = \{(1,1),(2,3),(3,2),(4,4),(5,3),(6,5)\}` で学習。
      パラメータは :math:`a = x_1`,  :math:`b = x_2`,  :math:`\log\sigma^2 = x_3` です（:math:`N = 3` のみ）。

      .. math::

          f(a, b, s) = \frac{1}{2}\left[n\,s + e^{-s}\sum_{k=1}^{n}(a\,t_k + b - y_k)^2\right]
          \quad (s = \log\sigma^2,\; n = 6)

    - 大域最適値: :math:`f(\vec{x}^*) \approx 1.005071 \quad (\vec{x}^* \approx (0.628571,\,0.8,\,-0.664976))`
