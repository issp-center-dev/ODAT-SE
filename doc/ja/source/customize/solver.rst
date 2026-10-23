.. _customize_solver:

``Solver`` の定義
================================

順問題を記述する ``Solver`` は、入力変数に対して目的関数の値を返す ``evaluate`` メソッドを持つクラスとして以下のように定義します。

- ``Solver`` クラスは ``odatse.solver.SolverBase`` を継承するクラスとします。

  .. code-block:: python

     import odatse

     class Solver(odatse.solver.SolverBase):
         pass

- コンストラクタ

  コンストラクタは ``Info`` クラスのインスタンスを引数としてとります。

  .. code-block:: python

     def __init__(self, info: odatse.Info):
         super().__init__(info)
         ...

  必ず ``info`` を引数として基底クラスのコンストラクタを呼び出してください。
  基底クラスのコンストラクタでは、次のインスタンス変数が設定されます。

  - ``self.root_dir`` はルートディレクトリです。 ``info.base["root_dir"]`` から取得され、 ``odatse`` を実行するディレクトリになります。外部プログラムやデータファイルなどを参照する際に起点として利用できます。

  - ``self.output_dir`` は出力ファイルを書き出すディレクトリです。 ``info.base["output_dir"]`` から取得されます。通例、MPI並列の場合は各ランクからのデータを集約した結果を出力します。

  - ``self.proc_dir`` はプロセスごとの作業用ディレクトリです。 ``output_dir / str(odatse.mpi.algrank())`` が設定されます。
    ソルバーの ``evaluate`` メソッドは ``proc_dir`` をカレントディレクトリとして Runner から呼び出され、MPIプロセスごとの中間結果などを出力します。
    MPIを使用しない場合もランク番号を0として扱います。

  - ``self.work_dir`` は ``self.proc_dir`` の別名です。

  - ``self.dimension`` は入力変数の次元数です。 ``info.solver`` に ``dimension`` が指定されていればその値、なければ ``info.base["dimension"]`` が設定されます。

  - ``self.timer`` は実行時間を記録するための辞書で、 ``"prepare"``, ``"run"``, ``"post"`` のキーを持ちます。

  - ``self._name`` はソルバー名（文字列）です。基底クラスでは空文字列に初期化されるため、コンストラクタで適切な名前を設定してください。 ``name`` プロパティを通して参照されます。

  Solver 固有のパラメータは ``info`` の ``solver`` フィールドから取得します。必要な設定を読み取って保存します。


- ``evaluate`` メソッド

  .. code-block:: python

         def evaluate(self, x, args=()) -> float:
             pass

  入力変数に対して目的関数の値を返すメソッドです。以下の引数を取ります。

  - ``x: np.ndarray``

    入力変数を numpy.ndarray 型の :math:`N` 次元ベクトルとして受け取ります。

  - ``args: Tuple = ()``

    Algorithm から渡される追加の引数で、step数と set番号からなる Tuple です。step数は Monte Carlo のステップ数や、グリッド探索のグリッド点のインデックスです。set番号は n巡目を表します。

  ``evaluate`` メソッドは、Float 型の目的関数の値を返します。

  .. note::
     ``evaluate`` が ``RuntimeError`` を送出した場合、 ``[runner]`` セクションで ``ignore_error = true`` が指定されていると、Runner は例外を無視して目的関数値を ``np.nan`` として扱います。
     また、探索点が制約条件 (``[runner.limitation]``) を満たさない場合はソルバーは呼び出されず、目的関数値は ``np.inf`` になります。

  .. note::
     ソルバー並列を使用する場合 (``--nsolve`` が 2 以上)、 ``evaluate`` はソルバーグループ内の全 MPI ランクで同一の ``x``, ``args`` を引数として呼び出されます。
     ランク間の役割分担はソルバー内で実装してください。詳細は :doc:`../tutorial/parallel_solver` を参照してください。
     ソルバーグループ内のどのランクで ``evaluate`` が例外を送出しても、コントローラランク (``solrank() == 0``) に報告されてそこで処理されます。失敗した全ランクの例外が ``RuntimeError`` であれば ``ignore_error`` が適用され、それ以外の場合はジョブはエラー終了します。ランクが失敗した場合でも、 ``evaluate`` 内の集団通信はグループ全体で揃っている必要があります。

- 勾配への対応 (任意)

  目的関数の勾配が計算できる場合、次のメンバーを実装すると、 ``Algorithm`` が勾配を利用できるようになります (例: ``minsearch`` での ``use_gradient = true``)。

  - ``supports_gradient`` (プロパティ、デフォルト: False)

    このソルバーが勾配を計算できるとき、 True を返すようにオーバーライドします。

  - ``gradient(self, x, args=()) -> np.ndarray``

    入力 ``x`` における目的関数の勾配を、 ``x`` と同じ shape で返します。
    Runner は戻り値を float の ndarray に変換し、shape が ``x`` と異なる場合は
    ``ignore_error = true`` でも ``ValueError`` を送出します。
    ``supports_gradient`` が True のときのみ呼ばれます。基底クラスでは ``NotImplementedError`` を送出します。

    ``[runner]`` で ``ignore_error = true`` を指定した場合、 ``gradient()`` が送出した
    ``RuntimeError`` は無視され、Runner は ``x`` と同じ shape で全要素が NaN の配列を返します。
    ただし、 ``NotImplementedError`` は ``ignore_error = true`` でも無視されず、そのまま送出されます。
    ``ValueError`` や ``TypeError`` など、その他の例外もそのまま送出されます。

  ``runner.log`` が数えるのは ``evaluate()`` の呼び出しだけです。勾配の呼び出しは記録されず、回数にも含まれません。

  ``[runner.mapping]`` を指定した場合と、ソルバー並列 (``--nsolve`` が 2 以上) の場合は、勾配の利用に対応していません。
