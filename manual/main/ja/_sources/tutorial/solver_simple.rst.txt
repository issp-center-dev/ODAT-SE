順問題ソルバーの追加
======================

ベンチマーク関数ソルバー
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``py2dmat`` では探索アルゴリズムのテストに利用できる順問題ソルバーとして、 ``analytical`` ソルバーを準備しています。

``analytical`` ソルバーを使うには、入力ファイルの ``[solver]`` セクションの ``name`` を ``"analytical"`` に設定します。
また、 ``function_name`` パラメータを用いてベンチマーク関数 :math:`f(x)` を選択します。
たとえば、 Himmelblau 関数を用いる場合には ::

    [solver]
    name = "analytical"
    function_name = "himmelblau"

とします。
利用可能な関数の詳細は :doc:`analytical ソルバーのリファレンス <../solver/analytical>` を参照してください。

順問題ソルバーの追加
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ユーザ定義の順問題ソルバーを定義・解析する一番簡単な方法は、 ``analytical`` ソルバーに追加することです。
ここでは例として、 `Booth関数 <https://en.wikipedia.org/wiki/Test_functions_for_optimization>`_

.. math::

   f(x,y) = (x+2y-7)^{2} + (2x+y-5)^{2}

を追加してみましょう（最小値は :math:`f(1,3) = 0` ）。

そのためには、 ``py2dmat`` のソースコードをダウンロードし、ファイルを編集する必要があります。
ダウンロード方法や、ソースコードからの実行方法などは、 :doc:`インストールページ <../start>` を参照してください。

``analytical`` ソルバーは ``src/py2dmat/solver/analytical.py`` に定義されているので、これを編集します。

まず、 ``booth`` 関数を定義します。

.. code-block:: python

    def booth(xs: np.ndarray) -> float:
        """Booth function

        it has one global minimum f(xs) = 0 at xs = [1,3].
        """

        if xs.shape[0] != 2:
            raise RuntimeError(
                f"ERROR: booth expects d=2 input, but receives d={xs.shape[0]} one"
            )
        return (xs[0] + 2 * xs[1] - 7.0) ** 2 + (2 * xs[0] + xs[1] - 5.0) ** 2

つぎに、入力ファイルの ``solver.function_name`` パラメータで ``booth`` 関数を指定できるようにするために、
``Solver`` クラスのコンストラクタ (``__init__``) 中の if 分岐に以下のコードを挿入します。

.. code-block:: python

    elif function_name == "booth":
            self.set_function(booth)

この改造した ``analytical`` ソルバーでは、 Booth 関数の最適化が行なえます。
たとえばNelder-Mead 法による最適化は、以下の入力ファイル (``input.toml``) を ::

    [base]
    dimension = 2
    output_dir = "output"

    [algorithm]
    name = "minsearch"
    seed = 12345

    [algorithm.param]
    max_list = [6.0, 6.0]
    min_list = [-6.0, -6.0]
    initial_list = [0, 0]

    [solver]
    name = "analytical"
    function_name = "booth"

``src/py2dmat_main.py`` に渡せば実行可能です::

    $ python3 src/py2dmat_main.py input.toml

    ... skipped ...

    Iterations: 38
    Function evaluations: 75
    Solution:
    x1 = 1.0000128043523089
    x2 = 2.9999832920260863
