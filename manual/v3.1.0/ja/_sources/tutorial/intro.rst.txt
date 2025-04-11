解析関数の最小化問題
================================

順問題ソルバーの例として ODAT-SE に含まれている Analytical ソルバーの中から Himmelblau関数の最小化問題を取り上げます。
Himmelblau関数は次の表式で表される2変数関数で、複数の極小点を持ち、最適化アルゴリズムの性能評価に使われます。

.. math::

   f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2

最小値 :math:`f(x,y)=0` を与える :math:`(x,y)` は :math:`(3.0, 2.0)`, :math:`(-2.805118, 3.131312)`, :math:`(-3.779310, -3.283186)`, :math:`(3.584428, -1.848126)` です。

.. figure:: ../../../common/img/plot_himmelblau.*

  Himmelblau関数の plot。


[1] D. Himmelblau, Applied Nonlinear Programming, McGraw-Hill, 1972.

