``Solver`` の定義
================================

順問題を記述する ``Solver`` は、入力変数に対して目的関数の値を返す ``evaluate`` メソッドを持つクラスとして以下のように定義します。

- ``Solver`` クラスは ``odatse.solver.SolverBase`` を継承するクラスとします。

  .. code-block:: python

     import odatse

     class Solver(odatse.solver.SolverBase):
         pass

- コンストラクタ

  コンストラクタは ``Info`` クラスのメソッドを引数としてとります。

  .. code-block:: python

     def __init__(self, info: odatse.Info):
         super().__init__(info)
	 ...

  必ず ``info`` を引数として基底クラスのコンストラクタを呼び出してください。
  基底クラスのコンストラクタでは、次のインスタンス変数が設定されます。

  - ``self.root_dir`` はルートディレクトリです。 ``info.base["root_dir"]`` から取得され、 ``odatse`` を実行するディレクトリになります。外部プログラムやデータファイルなどを参照する際に起点として利用できます。

  - ``self.output_dir`` は出力ファイルを書き出すディレクトリです。 ``info.base["output_dir"]`` から取得されます。通例、MPI並列の場合は各ランクからのデータを集約した結果を出力します。

  - ``self.proc_dir`` はプロセスごとの作業用ディレクトリです。 ``output_dir / str(self.mpirank)`` が設定されます。
    ソルバーの ``evaluate`` メソッドは ``proc_dir`` をカレントディレクトリとして Runner から呼び出され、MPIプロセスごとの中間結果などを出力します。
    MPIを使用しない場合もランク番号を0として扱います。

  Solver 固有のパラメータは ``info`` の ``solver`` フィールドから取得します。必要な設定を読み取って保存します。

    
- ``evaluate`` メソッド  

  .. code-block:: python

         def evaluate(self, x, args=(), nprocs=1, nthreads=1) -> float:
	     pass

  入力変数に対して目的変数の値を返すメソッドです。以下の引数を取ります。

  - ``x: np.ndarray``

    入力変数を numpy.ndarray 型の :math:`N` 次元ベクトルとして受け取ります。

  - ``args: Tuple = ()``

    Algorithm から渡される追加の引数で、step数と set番号からなる Tuple です。step数は Monte Carlo のステップ数や、グリッド探索のグリッド点のインデックスです。set番号は n巡目を表します。

  - ``nprocs: int = 1``

  - ``nthreads: int = 1``

    ソルバーを MPI並列・スレッド並列で実行する際のプロセス数・スレッド数を受け取ります。現在は ``procs=1``, ``nthreads=1`` のみ対応しています。

  ``evaluate`` メソッドは、Float 型の目的関数の値を返します。
