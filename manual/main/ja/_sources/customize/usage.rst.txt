実行方法
================================

次のようなフローで最適化問題を実行できます。
プログラム例にあるコメントの番号はフローの番号に対応しています。

1. ユーザ定義クラスを作成する

   - ODAT-SEで定義済みのクラスも利用可能です

2. 入力パラメータ ``info: odatse.Info`` を作成する

   - ``Info`` クラスにはTOML形式の入力ファイルを読み込むクラスメソッドが用意されています。この他にも、dict形式でパラメータを用意して ``Info`` クラスのコンストラクタに渡して作成することができます。

3. ``solver: Solver``, ``runner: odatse.Runner``, ``algorithm: Algorithm`` を作成する

4. ``algorithm.main()`` を実行する


プログラム例 

.. code-block:: python

    import sys
    import odatse

    # (1)
    class Solver(odatse.solver.SolverBase):
        # Define your solver
        pass

    class Algorithm(odatse.algorithm.AlgorithmBase):
        # Define your algorithm
        pass

    # (2)
    input_file = sys.argv[1]
    info = odatse.Info.from_file(input_file)

    # (3)
    solver = Solver(info)
    runner = odatse.Runner(solver, info)
    algorithm = Algorithm(info, runner)

    # (4)
    result = algorithm.main()
