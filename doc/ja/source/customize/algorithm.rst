``Algorithm`` の定義
================================

``Algorithm`` クラスは ``odatse.algorithm.AlgorithmBase`` を継承したクラスとして定義します。

.. code-block:: python

    import odatse

    class Algorithm(odatse.algorithm.AlgorithmBase):
        pass


``AlgorithmBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlgorithmBase`` クラスは次のメソッドを提供します。

- ``__init__(self, info: odatse.Info, runner: odatse.Runner = None)``

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

	- ``self.checkpoint`` : チェックポイント機能を有効/無効にする

	    - ``self.checkpoint_steps``
	    - ``self.checkpoint_interval``
	    - ``self.checkpoint_file``

	      チェックポイント機能に関するパラメータが設定されます

	      
- ``prepare(self) -> None``

    - 最適化アルゴリズムの前処理をします
    - ``self.run()`` の前に実行する必要があります

- ``run(self) -> None``

    - 最適化アルゴリズムを実行します
    - 以下の処理を行います

      #. ``self.proc_dir`` に移動する
      #. ``self.runner.prepare()`` を実行する
      #. ``self._run()`` を実行する
      #. ``self.runner.post()`` を実行する
      #. 元のディレクトリに戻る
    - ``self.prepare()`` の後に実行する必要があります

- ``post(self) -> Dict``

    - 最適化結果のファイル出力など、後処理を行います
    - ``self.output_dir`` に移動し、 ``self._post()`` を実行した後、元のディレクトリに戻ります
    - ``self.run()`` のあとに実行する必要があります

- ``main(self, run_mode) -> Dict``

    - ``prepare``, ``run``, ``post`` を順番に実行します
    - それぞれの関数でかかった時間を計測し、結果をファイル出力します
    - 実行モードを文字列型の引数で受け取ります。デフォルト値は ``initialize`` です。

      - ``"initialize"``: 最初から実行
      - ``"resume"``: 中断した状態から再実行
      - ``"continue"``: 終了時の状態から継続実行

      乱数の初期化を指示する場合は、 ``"-resetrand"`` が追加されます。
      ``"continue"`` の動作はアルゴリズムによって変わります。
      
    - 探索の結果を辞書形式で返します



``Algorithm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Algorithm`` は少なくとも次のメソッドを定義しなければなりません。

- ``__init__(self, info: odatse.Info, runner: odatse.Runner = None, domain = None)``

    - ``info`` および ``runner`` 引数はそのまま基底クラスのコンストラクタに転送してください

        - ``super().__init__(info=info, runner=runner)``

    - 入力パラメータである ``info`` から必要な設定を読み取り、保存してください

    - ``domain`` が指定されている場合は、探索領域を ``domain`` から取得します。
      指定されていない場合は ``odatse.domain.Region(info)`` (探索領域が連続的な場合) または ``odatse.domain.MeshGrid(info)`` (離散的な場合) を用いて ``info`` から作成します。

- ``_prepare(self) -> None``

    - 最適化アルゴリズムの前処理を記述します

- ``_run(self) -> None``

    - 最適化アルゴリズムを記述します
    - 探索パラメータ ``x`` から対応する目的関数の値 ``f(x)`` を得るには、次のように Runner クラスのメソッドを呼び出します。

      .. code-block:: python

	 args = (step, set)
         fx = self.runner.submit(x, args)

- ``_post(self) -> Dict``

    - 最適化アルゴリズムの後処理を記述します
    - 探索の結果を辞書形式で返します
      

``Domain`` の定義
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

探索領域を記述する 2種類のクラスが用意されています。

``Region`` クラス
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

連続的なパラメータ空間を定義するためのヘルパークラスです。

- コンストラクタ引数は ``Info`` または ``param=`` にdict形式のパラメータを取ります。

  - ``Info`` 型の引数の場合、 ``Info.algorithm.param`` から探索範囲の最小値・最大値・単位や初期値を取得します。

  - dict 型の引数の場合は ``Info.algorithm.param`` 相当の内容を辞書形式で受け取ります。

  - 詳細は :ref:`min_search の入力ファイル <minsearch_input_param>` を参照してください。

- ``initialize(self, rng, limitation, num_walkers)`` を呼んで初期値の設定を行います。引数は乱数発生器 ``rng``, 制約条件 ``limitation``, walker の数 ``num_walkers`` です。


``MeshGrid`` クラス
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

離散的的なパラメータ空間を定義するためのヘルパークラスです。

- コンストラクタ引数は ``Info`` または ``param=`` にdict形式のパラメータを取ります。

  - ``Info`` 型の引数の場合、 ``Info.algorithm.param`` から探索範囲の最小値・最大値・単位や初期値を取得します。

  - dict 型の引数の場合は ``Info.algorithm.param`` 相当の内容を辞書形式で受け取ります。

  - 詳細は :ref:`mapper の入力ファイル <mapper_input_param>` を参照してください

- ``do_split(self)`` メソッドは、候補点の集合を分割して各MPIランクに配分します。

- 入出力について

  - ``from_file(cls, path)`` クラスメソッドは、 ``path`` からメッシュデータを読み込んで ``MeshGrid`` クラスのインスタンスを作成します。

  - ``store_file(self, path)`` メソッドは、メッシュの情報を ``path`` のファイルに書き出します。
