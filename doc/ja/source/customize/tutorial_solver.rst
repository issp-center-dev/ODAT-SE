チュートリアル: 独自ソルバーの追加
==========================================

このチュートリアルでは、独自の目的関数を定義して ODAT-SE の探索アルゴリズムで最小化する方法を、
ステップごとに説明します。

ここでは例として、次の2変数関数の最小値を Nelder-Mead 法で求めます。

.. math::

   f(x, y) = (x - 3)^2 + (y - 2)^2

最小値は :math:`f(3, 2) = 0` です。

前提条件
~~~~~~~~~~~~~~~~~~~~~~~~~

- ODAT-SE がインストール済みであること（ :doc:`/start` を参照）
- Python の基本的な文法を理解していること（関数定義、クラスの書き方）


全体の流れ
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Python スクリプトにソルバーを定義する
2. TOML 設定ファイルを作成する
3. 実行して結果を確認する


Step 1: ソルバーを定義する
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下の内容で ``my_solver.py`` というファイルを作成してください。

.. code-block:: python

    import sys
    import numpy as np
    import odatse

    # --- ソルバーの定義 ---
    class MySolver(odatse.solver.SolverBase):
        """自分の目的関数を計算するソルバー"""

        def __init__(self, info: odatse.Info):
            super().__init__(info)
            self._name = "my_solver"

            # TOML ファイルの [solver] セクションからパラメータを読み取れます
            # 例: self.param = info.solver.get("my_param", 1.0)

        def evaluate(self, x, args=()):
            """
            目的関数を計算して返す。

            Parameters
            ----------
            x : np.ndarray
                探索パラメータ（ここでは [x, y] の2次元ベクトル）
            args : tuple
                (step番号, set番号) のタプル。ログ出力などに利用可能。
            """
            # ここに自分の目的関数を書く
            fx = (x[0] - 3.0) ** 2 + (x[1] - 2.0) ** 2
            return fx

    # --- メインの実行コード ---
    # TOML ファイルのパスをコマンドライン引数から取得
    input_file = sys.argv[1]
    info = odatse.Info.from_file(input_file)

    # MPI コミュニケータを分割する。ソルバー/アルゴリズムは構築時に MPI 層
    # (algrank() など)を参照するため、構築前に呼ぶ必要がある。
    odatse.mpi.setup()

    # ソルバー → Runner → Algorithm の順に組み立てる
    solver = MySolver(info)
    runner = odatse.Runner(solver, info)
    alg_module = odatse.algorithm.choose_algorithm(info.algorithm["name"])
    algorithm = alg_module.Algorithm(info, runner)

    # 実行
    result = algorithm.main()

**ポイント:**

- ``MySolver`` は ``odatse.solver.SolverBase`` を継承しています
- ``__init__`` で ``super().__init__(info)`` を必ず呼びます。これにより出力ディレクトリなどが自動設定されます
- ``evaluate`` メソッドに目的関数の計算を書きます。引数 ``x`` は numpy 配列で、戻り値は float です
- パイプラインを手動で組み立てる場合、ソルバー/アルゴリズムを構築する前に ``odatse.mpi.setup()`` を呼ぶ必要があります( ``odatse.initialize()`` を使う場合は内部で呼ばれます)
- ``odatse.algorithm.choose_algorithm`` は TOML ファイルの ``[algorithm]`` セクションの名前に対応するアルゴリズムモジュールを返します。そのモジュールの ``Algorithm`` クラスをインスタンス化します


Step 2: TOML 設定ファイルを作成する
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下の内容で ``input.toml`` を作成してください。

.. code-block:: toml

    [base]
    dimension = 2
    output_dir = "output"

    [solver]
    name = "my_solver"
    # ここに solver 固有のパラメータを追加できます
    # my_param = 1.0

    [algorithm]
    name = "minsearch"
    seed = 12345

    [algorithm.param]
    max_list = [6.0, 6.0]
    min_list = [-6.0, -6.0]
    initial_list = [0.0, 0.0]

**各セクションの意味:**

- ``[base]``: ``dimension = 2`` は探索パラメータが2つ (x, y) であることを指定
- ``[solver]``: ソルバーの設定。``name`` はログ出力用で、任意の文字列を指定可能
- ``[algorithm]``: 探索アルゴリズムの設定。``minsearch`` は Nelder-Mead 法
- ``[algorithm.param]``: 探索範囲の最大値・最小値と初期値


Step 3: 実行して結果を確認する
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``my_solver.py`` と ``input.toml`` を同じディレクトリに置いて、以下を実行します。

.. code-block:: bash

    $ python3 my_solver.py input.toml

実行が完了すると、以下のような出力が表示されます。

.. code-block:: text

    Iterations: 43
    Function evaluations: 82
    Solution:
    x1 = 2.9999999...
    x2 = 1.9999999...

パラメータが :math:`(x, y) = (3, 2)` に収束し、最小値が見つかったことがわかります。

また、 ``output/`` ディレクトリ以下に実行ログが出力されます。


応用: 他のアルゴリズムで探索する
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TOML ファイルの ``[algorithm]`` セクションを変更するだけで、Python コードを変えずに別のアルゴリズムで探索できます。

**ベイズ最適化の例:**

.. code-block:: toml

    [algorithm]
    name = "bayes"
    seed = 12345

    [algorithm.param]
    max_list = [6.0, 6.0]
    min_list = [-6.0, -6.0]

    [algorithm.bayes]
    random_max_num_probes = 10
    score = "TS"
    num_search_each_probe = 1

**グリッド探索の例:**

.. code-block:: toml

    [algorithm]
    name = "mapper"
    seed = 12345

    [algorithm.param]
    max_list = [6.0, 6.0]
    min_list = [-6.0, -6.0]
    num_list = [31, 31]


応用: TOML からソルバーのパラメータを読み取る
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

目的関数にパラメータを持たせたい場合は、``[solver]`` セクションに値を追加し、
``__init__`` で読み取ります。

.. code-block:: toml

    [solver]
    name = "my_solver"
    center_x = 5.0
    center_y = 3.0

.. code-block:: python

    class MySolver(odatse.solver.SolverBase):
        def __init__(self, info: odatse.Info):
            super().__init__(info)
            self._name = "my_solver"
            # TOML の [solver] セクションから値を取得
            self.cx = info.solver.get("center_x", 0.0)
            self.cy = info.solver.get("center_y", 0.0)

        def evaluate(self, x, args=()):
            return (x[0] - self.cx) ** 2 + (x[1] - self.cy) ** 2

このように、TOML ファイルとソルバーの組み合わせで、コードを変更せずにパラメータを変えて実験できます。


応用: ソルバーテンプレートを使う
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ソルバーをパッケージとして整備したい場合は、`ODAT-SE ソルバーテンプレート <https://isspns-gitlab.issp.u-tokyo.ac.jp/takeohoshi/odat-se-gallery/-/tree/main/data/tutorial/solver-template>`_ を利用すると便利です。
用途に応じた4種類のテンプレートが用意されています。

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - テンプレート
     - 用途
     - 特徴
   * - ``user_function``
     - Python 関数を手軽に最適化したい
     - 最小構成のスクリプト。Step 1〜3 の内容に近い
   * - ``function_module``
     - 解析関数ソルバーをパッケージ化したい
     - ``pip install`` 可能なモジュール。関数を追加するだけ
   * - ``solver_module``
     - データファイルを読み込む独自ソルバーを作りたい
     - 参照データとの比較（尤度計算等）を行うソルバーの雛形
   * - ``external_solver_module``
     - 外部プログラム（C/Fortran等）をソルバーとして使いたい
     - 入出力ファイル管理、subprocess 実行、作業ディレクトリ管理を含む

**使い方の例（function_module の場合）:**

1. テンプレートをコピーする

   .. code-block:: bash

       $ cp -r odat-se-gallery/data/tutorial/solver-template/function_module my_solver_pkg
       $ cd my_solver_pkg

2. ``pyproject.toml`` のパッケージ名を変更し、``src/Solver/`` 以下にソルバーを追加する

3. インストールして実行する

   .. code-block:: bash

       $ python3 -m pip install .

各テンプレートにはサンプル設定ファイル（``sample/``）やテストの雛形（``tests/``）も含まれているので、
本格的なソルバーパッケージの開発の出発点として活用できます。
詳細はテンプレート内の ``docs/`` ディレクトリを参照してください。


次のステップ
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``evaluate`` メソッドの中で外部プログラムを呼び出すことで、より複雑な順問題ソルバーと連携できます
- ソルバーの API 詳細は :doc:`solver` を参照してください
- 独自の探索アルゴリズムを定義する場合は :doc:`algorithm` を参照してください
