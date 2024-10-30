``Algorithm`` の定義
======================

``Algorithm`` クラスは ``py2dmat.algorithm.AlgorithmBase`` を継承したクラスとして定義します。 ::

    import py2dmat

    class Algorithm(py2dmat.algorithm.AlgorithmBase):
        pass

``AlgorithmBase``
~~~~~~~~~~~~~~~~~~~

``AlgorithmBase`` クラスは次のメソッドを提供します。

- ``__init__(self, info: py2dmat.Info, runner: py2dmat.Runner = None)``

    - ``info`` から ``Algorithm`` 共通の入力パラメータを読み取り、次のインスタンス変数を設定します。

        - ``self.mpicomm: Optional[MPI.Comm]`` : ``MPI.COMM_WORLD``

            - ``mpi4py`` の import に失敗した場合、 ``None`` が設定されます

        - ``self.mpisize: int`` : MPIプロセス数

            - ``mpi4py`` の import に失敗した場合、 ``1`` が設定されます

        - ``self.mpirank: int`` : MPIランク

            - ``mpi4py`` の import に失敗した場合、 ``0`` が設定されます

        - ``self.rng: np.random.Generator`` : 擬似乱数生成器

            - 擬似乱数の種について、詳細は :ref:`入力パラメータの [algorithm] セクション <input_algorithm>` を参照してください

        - ``self.dimension: int`` : 探索パラメータ空間の次元
        - ``self.label_list: List[str]`` : 各パラメータの名前
        - ``self.root_dir: pathlib.Path`` : ルートディレクトリ

            - ``info.base["root_dir"]``

        - ``self.output_dir: pathlib.Path`` : 出力ファイルを書き出すディレクトリ

            - ``info.base["root_dir"]``

        - ``self.proc_dir: pathlib.Path`` : プロセスごとの作業用ディレクトリ

            - ``self.output_dir / str(self.mpirank)``
            - ディレクトリが存在しない場合、自動的に作成されます
            - 各プロセスで最適化アルゴリズムはこのディレクトリで実行されます

        - ``self.timer: Dict[str, Dict]`` : 実行時間を保存するための辞書

            - 空の辞書が3つ、 ``"prepare"``, ``"run"``, ``"post"`` という名前で定義されます

- ``prepare(self) -> None``

    - 最適化アルゴリズムの前処理をします
    - ``self.run()`` の前に実行する必要があります

- ``run(self) -> None``

    - 最適化アルゴリズムを実行します
    - ``self.proc_dir`` に移動し、 ``self._run()`` を実行した後、元のディレクトリに戻ります

- ``post(self) -> None``

    - 最適化結果のファイル出力など、後処理を行います
    - ``self.output_dir`` に移動し、 ``self._post()`` を実行した後、元のディレクトリに戻ります
    - ``self.run()`` のあとに実行する必要があります

- ``main(self) -> None``

    - ``prepare``, ``run``, ``post`` を順番に実行します
    - それぞれの関数でかかった時間を計測し、結果をファイル出力します


- ``_read_param(self, info: py2dmat.Info) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]``

    - 連続なパラメータ空間を定義するためのヘルパーメソッドです
    - ``info.algorithm["param"]`` から探索パラメータの初期値や最小値、最大値、単位を取得します
    - 詳細は :ref:`min_search の入力ファイル <minsearch_input_param>` を参照してください

- ``_mesh_grid(self, info: py2dmat.Info, split: bool = False) -> Tuple[np.ndarray, np.ndarray]``

    - 離散的なパラメータ空間を定義するためのヘルパーメソッドです
    - ``info.algorithm["param"]`` を読み取り次を返します:

        - ``D`` 次元の候補点 ``N`` 個からなる集合 (``NxD`` 次元の行列として)
        - ``N`` 個の候補点のID(index)

    - ``split`` が ``True`` の場合、候補点集合は分割され各MPI ランクに配られます
    - 詳細は :ref:`mapper の入力ファイル <mapper_input_param>` を参照してください


``Algorithm``
~~~~~~~~~~~~~~~~

``Algorithm`` は少なくとも次のメソッドを定義しなければなりません。

- ``__init__(self, info: py2dmat.Info, runner: py2dmat.Runner = None)``

    - 引数はそのまま基底クラスのコンストラクタに転送してください

        - ``super().__init__(info=info, runner=runner)``

    - 入力パラメータである ``info`` から必要な設定を読み取り、保存してください

- ``_prepare(self) -> None``

    - 最適化アルゴリズムの前処理を記述します

- ``_run(self) -> None``

    - 最適化アルゴリズムを記述します
    - 探索パラメータ ``x`` から対応する目的関数の値 ``f(x)`` を得る方法 ::

        message = py2dmat.Message(x, step, set)
        fx = self.runner.submit(message)

- ``_post(self) -> None``

    - 最適化アルゴリズムの後処理を記述します
