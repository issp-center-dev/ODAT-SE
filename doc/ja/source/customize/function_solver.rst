.. _customize_function_solver:

``function.Solver`` による定義
================================

目的関数が Python の関数 ``f(x) -> float`` として書ける場合は、
``SolverBase`` を直接継承する代わりに ``odatse.solver.function.Solver`` を使うと、
``evaluate`` メソッドを実装せずにソルバーを作れます。
組み込みの ``analytical`` ソルバー (:doc:`../solver/analytical`) もこのクラスの派生クラスです。

使い方
~~~~~~~~~~~~~~~~~~~~~~~~~

インスタンスを作り、 ``set_function()`` で目的関数を渡します。

.. code-block:: python

    import numpy as np
    import odatse
    import odatse.solver.function

    def f(x: np.ndarray) -> float:
        return np.sum((x - 1.0) ** 2)

    solver = odatse.solver.function.Solver(info)
    solver.set_function(f)

``sample/user_function/simple.py`` に、このソルバーで探索を実行する例があります。

派生クラスを作り、コンストラクタで関数を設定することもできます。

.. code-block:: python

    class Solver(odatse.solver.function.Solver):
        def __init__(self, info: odatse.Info):
            super().__init__(info)
            self._name = "my_function"
            self.set_function(f)

メソッドと属性
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``set_function(self, f) -> None``

  目的関数 ``f(x: np.ndarray) -> float`` を設定します。
  設定済みの勾配は解除されます。勾配を使う場合は、この後に ``set_gradient()`` を呼んでください。

- ``set_gradient(self, g) -> None``

  目的関数の勾配 ``g(x: np.ndarray) -> np.ndarray`` を設定します。
  ``g`` は ``x`` と同じ shape の配列を返す必要があります。
  勾配を設定すると、 ``supports_gradient`` は自動的に True になり、
  ``minsearch`` の ``use_gradient = true`` などで勾配を使えます。

- ``evaluate(self, x, args=()) -> float``

  ``work_dir`` をカレントディレクトリにして ``f(x)`` を呼び、その値を返します。
  ``args`` は ``f`` に渡しません。
  関数が設定されていない場合は ``RuntimeError`` を送出します。

- ``gradient(self, x, args=()) -> np.ndarray``

  ``g(x)`` を呼び、その値を返します。 ``args`` は ``g`` に渡しません。
  ``evaluate`` と異なり、カレントディレクトリは変更せず、 ``delay`` も適用しません。
  勾配が設定されていない場合は ``NotImplementedError`` を送出します。

- ``supports_gradient``

  勾配が設定されているときに True になります。
  派生クラスで ``self.supports_gradient`` に値を代入した場合はその値が優先され、
  後で ``set_function()`` や ``set_gradient()`` を呼んでも変わりません。

入力パラメータ
~~~~~~~~~~~~~~~~~~~~~~~~~

``[solver]`` セクションで次のパラメータを指定できます。

- ``delay``

  形式: 実数 (default: 0.0)

  説明: デバッグ用のパラメータです。 ``evaluate`` のたびに、関数の評価後にこの秒数だけ待ちます。
