ソルバーの二層 MPI 並列化
==========================================

はじめに
--------

このチュートリアルでは、ODAT-SE フレームワークの中で、二層の MPI 並列を活用するカスタムソルバーの書き方を説明します。サンプルファイルはリポジトリのルートからの相対パス ``sample/parallel_solver`` にあり、メインスクリプトは ``parallel_solver.py`` です。

2つの並列化レベル
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ODAT-SE のワークフローは2つの段階からなります。探索 *アルゴリズム* がパラメータ空間中の候補点を提案し、 *ソルバー* がその点で目的関数を評価します。ODAT-SE は、この両方の段階を MPI で同時に並列化できます。

- **アルゴリズム層の並列** (``nalg`` プロセス): 探索空間を、独立した評価に分散します。
- **ソルバー層の並列** (グループあたり ``nsolve`` プロセス): 1回の評価の計算を、プロセスのグループ内で分散します。

MPI プロセスの総数は ``nalg × nsolve`` です。

各層のプロセス数はコマンドラインで ``--nalg`` と ``--nsolve`` として与え、 ``odatse.initialize()`` がそれらを ``odatse.mpi.setup()`` へ渡します。例えば

.. code:: bash

    mpirun -np 6 python3 parallel_solver.py --nalg 3 --nsolve 2

は、3個のアルゴリズムプロセスとグループあたり2個のソルバープロセス、合計6 MPI ランクで実行します。

各ソルバープロセス内のスレッド並列(例えば NumPy が呼び出す BLAS ルーチン)は、環境変数 ``OMP_NUM_THREADS`` で別途制御します。ODAT-SE 自体はスレッドを管理しません。

.. note::

   ODAT-SE はソルバープロセスごとのスレッド数を制御しないため、これはライブラリの外で行う必要があります。通常は ``OMP_NUM_THREADS`` などの環境変数で設定します(例えばジョブ起動前に ``export OMP_NUM_THREADS=2`` )。あるいは、``threadpoolctl`` のようなライブラリを使ってソルバー内でプログラム的に設定することもできます。例えば重い計算を ``with threadpoolctl.threadpool_limits(limits=n_threads): ...`` で囲みます。

2つの層の設定方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ODAT-SE を MPI 下で実行すると、 ``odatse.mpi.setup(nalg=..., nsolve=...)`` がグローバルコミュニケータ(``MPI_COMM_WORLD``)を ``nalg`` 個のソルバーサブコミュニケータ(``solcomm``)に分割します。各 ``solcomm`` はそれぞれ ``nsolve`` プロセスを持ちます。各 ``solcomm`` でランク0のプロセスがグループの *コントローラ* となり探索アルゴリズムに参加します。コントローラ全体がアルゴリズムサブコミュニケータ(``algcomm``)を構成します。コントローラは各グループの最小ランクなので、グローバルランク0は常にコントローラです。

``odatse.mpi`` モジュールは以下のアクセサを提供します。グローバル層のものと ``enabled()`` はすぐに利用でき、ソルバー層・アルゴリズム層のものは ``setup()`` の呼び出し後に利用可能です。

- ``odatse.mpi.comm()`` / ``size()`` / ``rank()``: グローバルコミュニケータ、そのサイズ、およびこのプロセスのランク。
- ``odatse.mpi.solcomm()`` / ``solsize()`` / ``solrank()``: ソルバーサブコミュニケータ、そのサイズ(``nsolve``)、およびこのプロセスのランク。
- ``odatse.mpi.algcomm()`` / ``algsize()`` / ``algrank()``: アルゴリズムサブコミュニケータ、そのサイズ(``nalg``)、およびこのプロセスのランク。``algcomm()`` はソルバーワーカープロセスでは ``None`` を返します。``algsize()`` と ``algrank()`` はグループのコントローラの値(ワーカーへブロードキャストされる)を返すので、どのプロセスでもグループの識別に使えます。
- ``odatse.mpi.run_on_algorithm()``: グループのコントローラ(``solrank() == 0``)では ``True``、ソルバーワーカーでは ``False``。
- ``odatse.mpi.enabled()``: MPI が利用可能かどうか(環境変数 ``ODATSE_NOMPI=1`` が設定されている場合は ``False``)。

マスター・ワーカー実行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

各ソルバーグループ内では、マスター・ワーカー方式によりコントローラとワーカーが同期されます。これはフレームワークが完全に処理するため、ソルバーの作成者がワーカーループを書く必要はありません。

- コントローラ(``solrank() == 0``)は、通常どおりアルゴリズムの ``prepare()`` 、 ``run()`` 、 ``post()`` を実行します。アルゴリズムが候補 ``x`` を評価するたびに、 ``Runner.submit()`` が ``x`` と追加引数 ``args`` をソルバーグループ全体にブロードキャストし、その後にソルバーの ``evaluate(x, args)`` を呼び出します。
- ワーカー(``solrank() > 0``)は、フレームワーク内のループで待機し、ブロードキャストされた ``x`` / ``args`` を受け取り、同じ ``evaluate(x, args)`` を呼び出します。その戻り値は破棄されます。

言い換えると、 ``evaluate(x, args)`` は、ソルバーグループの **すべて** のプロセスで **同じ** ``x`` と ``args`` を用いて呼び出されます。ソルバー本体は、その1回の評価の計算を ``solcomm`` の集団通信を使ってグループ内で分散できます。アルゴリズムが終了すると、フレームワークがワーカーにループを抜けるよう通知します。

カスタムソルバーの例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

おもちゃの問題として、与えられた個数 ``nmats`` の ``matsize × matsize`` のランダム行列について、最大特異値の平均を最小化する整数シードを ``{1, …, 20}`` から探します。ソルバーは ``odatse.solver.SolverBase`` のサブクラスです(スクリプト全体は ``sample/parallel_solver/parallel_solver.py``)。

.. code:: python

    import os, time, argparse
    import numpy as np
    from mpi4py import MPI
    import odatse
    from odatse.algorithm import choose_algorithm

    class ParallelSolver(odatse.solver.SolverBase):
        def __init__(self, info, **kwargs):
            super().__init__(info)
            if not odatse.mpi.enabled():
                raise RuntimeError(
                    "This sample requires MPI (do not set ODATSE_NOMPI=1)"
                )

            self.opt_x = None
            self.opt_fx = np.inf

            self.nmats = kwargs["nmats"]
            self.matsize = kwargs["matsize"]

            if odatse.mpi.rank() == 0:
                print(f"nalg: {odatse.mpi.algsize()}")
                print(f"nsolve: {odatse.mpi.solsize()}")
            odatse.mpi.comm().barrier()

        def _testfunc(self, mats):
            return np.sum([np.max(np.linalg.svd(mat, compute_uv=False)) for mat in mats])

        def _compute(self, seeds):  # called by all solcomm ranks
            results = []
            for seed in seeds:
                if odatse.mpi.solrank() == 0:
                    prng = np.random.default_rng(seed=int(seed))
                    mats = [prng.random(size=(self.matsize, self.matsize)) for _ in range(self.nmats)]
                else:
                    mats = None
                mats = odatse.mpi.solcomm().bcast(mats, root=0)
                mats = np.array_split(mats, odatse.mpi.solsize())[odatse.mpi.solrank()]
                results.append(self._testfunc(mats))
            odatse.mpi.solcomm().barrier()
            results = odatse.mpi.solcomm().allreduce(np.asarray(results), op=MPI.SUM)
            results /= self.nmats
            return results

        def evaluate(self, xs, args):
            seeds = xs.astype(int)

            if odatse.mpi.solrank() == 0:
                print(f"algrank: {odatse.mpi.algrank()}, seeds: {list(seeds)}")

            results = self._compute(seeds)

            if odatse.mpi.solrank() == 0:
                print(f"algrank: {odatse.mpi.algrank()}, results: {list(results)}")

                best_x = np.argmin(results)
                best_fx = results[best_x]
                if best_fx < self.opt_fx:
                    self.opt_x = xs[best_x]
                    self.opt_fx = best_fx
            return results

目的関数の値は ``evaluate`` で計算されます。``evaluate`` はソルバーグループのすべてのランクで実行される点に注意してください。コントローラ(``solrank() == 0``)が各シードについて ``nmats`` 個のランダム行列を生成して ``solcomm`` 上でブロードキャストし、各ランクが ``_testfunc`` で自分の担当分の最大特異値を計算し、部分和がグループ内で集約されてから平均されます。実行中の暫定最良解を ``self.opt_x`` / ``self.opt_fx`` に記録するのはコントローラのみです。

ドライバーと入力ファイル
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

スクリプトは ``main()`` の中で ODAT-SE のパイプラインを組み立てます。``--nalg`` / ``--nsolve`` を解析し、それらを ``odatse.initialize()`` (内部で ``odatse.mpi.setup()`` を呼ぶ)に渡し、ソルバーとランナーを構築し、アルゴリズムを選択して実行します。``alg.main()`` が返った後、各ソルバーグループの最良結果を ``algcomm`` 上で集約します。

.. code:: python

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--nalg', help='# of processes for search algorithm', type=int, default=1)
        parser.add_argument('-n', '--nsolve', help='# of processes for solver', type=int, default=1)
        args = parser.parse_args()

        assert args.nalg * args.nsolve == odatse.mpi.comm().size

        argv = ["input.toml", "--init", f"--nalg={args.nalg}", f"--nsolve={args.nsolve}"]
        info, run_mode = odatse.initialize(argv)

        nmats = info.solver["param"].get("nmats", 50)
        matsize = info.solver["param"].get("matsize", 1000)

        output_dir = info.base.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)

        solver = ParallelSolver(info, nmats=nmats, matsize=matsize)
        runner = odatse.Runner(solver, info)
        alg_module = choose_algorithm(info.algorithm["name"])
        alg = alg_module.Algorithm(info, runner, run_mode=run_mode)
        result = alg.main()

        if odatse.mpi.run_on_algorithm():
            opt_fx, opt_x = min(
                odatse.mpi.algcomm().allgather((solver.opt_fx, solver.opt_x)),
                key=lambda x: x[0])

        if odatse.mpi.rank() == 0:
            print(f"\nopt_x={opt_x}")
            print(f"opt_fx={opt_fx}")

入力ファイル ``input.toml`` は ``mapper`` アルゴリズムを選択し、探索範囲とソルバーのパラメータを設定します。

.. code:: toml

    [base]
    dimension = 1
    output_dir = "output"

    [solver]
    name = "custom"

    [solver.param]
    nmats = 50
    matsize = 1000

    [algorithm]
    name = "mapper"

    [algorithm.param]
    min_list = [1]
    max_list = [20]
    num_list = [20]

ここで ``[solver.param]`` は ``main()`` で読み取るパラメータ(``nmats`` と ``matsize`` 、ランダム行列の個数とサイズ)を保持します。``[solver] name`` は名目上のものです。``parallel_solver.py`` は ``ParallelSolver`` を直接インスタンス化するため、この文字列は実行のラベルにすぎません。アルゴリズムが ``mapper`` なので、ODAT-SE は ``num_list = [20]`` のグリッド点(整数シード ``1`` から ``20``)を走査し、それらを ``nalg`` 個のソルバーグループのコントローラに分配します。

``nalg`` と ``nsolve`` の値は ``input.toml`` ではなくコマンドラインで渡します。

実行
~~~~

MPI プロセスの総数は ``nalg × nsolve`` と一致している必要があります。3個のアルゴリズムプロセスとグループあたり2個のソルバープロセス(合計6ランク)、各プロセス2 BLAS スレッドの場合は次のようにします。

.. code:: bash

    export OMP_NUM_THREADS=2
    mpirun -np 6 python3 parallel_solver.py --nalg 3 --nsolve 2

サンプルの ``do.sh`` はより小さな構成で実行します。

.. code:: bash

    export OMP_NUM_THREADS=2
    mpirun -np 4 python3 parallel_solver.py -m 2 -n 2

(``-m`` / ``-n`` は ``--nalg`` / ``--nsolve`` の短縮形です。) プロセスの活動を(``top`` などのツールで)見ると、ソルバー段階では各 MPI ``python`` プロセスが最大で ``OMP_NUM_THREADS × 100 %`` の CPU を使うのが確認できます。

このサンプルは MPI 必須です。 ``comm`` や ``solcomm`` の集団通信を使うため、
``ODATSE_NOMPI=1`` は設定しないでください(このモードでは ``odatse.mpi.enabled()``
が ``False`` になり、コミュニケータが利用できません)。1 プロセスだけでも
``mpirun`` / ``mpiexec`` で起動してください。

.. code:: bash

    mpirun -np 1 python3 parallel_solver.py --nalg 1 --nsolve 1
