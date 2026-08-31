実行方法
================================

次のようなフローで最適化問題を実行できます。
プログラム例にあるコメントの番号はフローの番号に対応しています。

1. ユーザー定義クラスを作成する

   - ODAT-SEで定義済みのクラスも利用可能です

2. 入力パラメータ ``info: odatse.Info`` を作成する

   - ``Info`` クラスにはTOML形式の入力ファイルを読み込むクラスメソッドが用意されています。この他にも、dict形式でパラメータを用意して ``Info`` クラスのコンストラクタに渡して作成することができます。

3. ``odatse.mpi.setup()`` を呼び出してから、``solver: Solver``, ``runner: odatse.Runner``, ``algorithm: Algorithm`` を作成する（``setup()`` は MPI コミュニケータを分割するため、ソルバー/アルゴリズムの構築前に呼ぶ必要があります。``odatse.initialize()`` を使う場合は内部で自動的に呼ばれます）

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
    odatse.mpi.setup()
    solver = Solver(info)
    runner = odatse.Runner(solver, info)
    algorithm = Algorithm(info, runner)

    # (4)
    result = algorithm.main()


コマンドライン引数を扱う場合
--------------------------------

``odatse`` コマンドと同じ引数体系（ ``--resume`` によるリスタートや ``--nalg`` / ``--nsolve`` による MPI 分割）を独自スクリプトでも利用する場合は、手順 (2), (3) の初期化を ``odatse.initialize()`` で行うのが便利です（ :doc:`common` 参照）。

.. code-block:: python

    import odatse

    # (1) ユーザー定義クラス（省略）

    # (2)(3) コマンドライン引数の解釈と初期化
    #        (odatse.mpi.setup() は内部で呼ばれます)
    info, run_mode = odatse.initialize()

    solver = Solver(info)
    runner = odatse.Runner(solver, info)
    algorithm = Algorithm(info, runner, run_mode=run_mode)

    # (4)
    result = algorithm.main()

``run_mode`` を ``Algorithm`` のコンストラクタに渡すことで、チェックポイントからの再開（ ``--resume`` ）や継続実行（ ``--cont`` ）が独自スクリプトでも機能します。
``sys.argv`` に依存したくない場合は、 ``odatse.initialize(["input.toml"])`` のように引数リストを明示的に渡してください。
