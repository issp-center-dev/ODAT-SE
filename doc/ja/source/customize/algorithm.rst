``Algorithm`` の定義
================================

``Algorithm`` クラスは ``odatse.algorithm.AlgorithmBase`` を継承したクラスとして定義します。

.. code-block:: python

  import odatse

  class Algorithm(odatse.algorithm.AlgorithmBase):
      pass


``AlgorithmBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlgorithmBase`` はすべてのアルゴリズムに共通のインフラを提供します。

ライフサイクル
^^^^^^^^^^^^^^

``main()`` はフレームワーク内部の wrapper メソッドを通じて3つのフェーズを駆動します。

.. code-block:: none

    main()
      ├── prepare()   runner.prepare() → dispatch(init/resume/continue) → _prepare()
      ├── run()       _run()
      └── post()      _post() → runner.post()

サブクラスが実装するのは *フック* — アンダースコアの **付いた** メソッド
``_initialize()``、``_prepare()``、``_run()``、``_post()`` です。
アンダースコアなしの ``prepare``、``run``、``post`` はフレームワーク内部の
wrapper であり、サブクラスでオーバーライドしては **いけません**。

``__init__`` が設定するインスタンス変数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``__init__(self, info: odatse.Info, runner: odatse.Runner = None)``

  ``info`` から共通の入力パラメータを読み取り、以下のインスタンス変数を設定します。

  - ``self.rng: np.random.RandomState`` : 擬似乱数生成器

  - ``self.label_list: list[str]`` : 各パラメータ軸の名前。

  - ``self.root_dir: pathlib.Path`` : ルートディレクトリ（``info.base["root_dir"]``）。

  - ``self.output_dir: pathlib.Path`` : 出力ディレクトリ（``info.base["output_dir"]``）。

  - ``self.proc_dir: pathlib.Path`` : プロセスごとの作業ディレクトリ。

    - ``self.output_dir / str(odatse.mpi.algrank())`` に設定されます。
    - 存在しない場合は自動的に作成されます。
    - ``_run()`` はこのディレクトリ内で呼び出されます。

  - ``self.timer: dict[str, dict]`` : 実行時間を保存するための辞書。

      - ``self.output_dir / str(odatse.mpi.algrank())``
      - ディレクトリが存在しない場合、自動的に作成されます
      - 各プロセスで最適化アルゴリズムはこのディレクトリで実行されます

    ``"prepare"``、``"run"``、``"post"`` の空の辞書が事前に作成されます。

  - ``self.checkpoint: bool`` : チェックポイント機能の有効/無効。
  - ``self.checkpoint_file: str`` : チェックポイントファイルの絶対パス（デフォルト: ``<proc_dir>/status.pickle``）。
  - ``self.checkpoint_steps: int`` : この ステップ数ごとにチェックポイントを保存します。
  - ``self.checkpoint_interval: float`` : この秒数ごとにチェックポイントを保存します。

  - ``self.mode: str`` : 実行モード文字列（``"initial"``、``"resume"``、``"continue"``、または ``"-resetrand"`` サフィックス付き）。

フレームワーク wrapper（オーバーライド禁止）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``prepare(self) -> None``

  ``runner.prepare()`` を呼び出した後、init/resume/continue を自動的にディスパッチし、``_prepare()`` を呼び出します。

  - ``mode`` が ``"init"`` で始まる場合 → ``_initialize()`` を呼び出します。
  - ``mode`` が ``"resume"`` または ``"continue"`` で始まる場合 → ``_load_state()`` を呼び出します。

  このメソッドをオーバーライドしないでください。代わりに ``_prepare()`` を実装してください。

- ``run(self) -> None``

  ``proc_dir`` に移動して ``_run()`` を呼び出します。
  Runner の呼び出しは ``prepare()`` と ``post()`` が担当します。

  このメソッドをオーバーライドしないでください。代わりに ``_run()`` を実装してください。

- ``post(self) -> dict``

  ``output_dir`` に移動し、``_post()`` を呼び出した後、``runner.post()`` を呼び出します。

  このメソッドをオーバーライドしないでください。代わりに ``_post()`` を実装してください。

- ``main(self) -> dict``

  ``prepare()``、``run()``、``post()`` を順に呼び出し、時間計測と MPI バリアを行います。
  最適化の結果を辞書形式で返します。

チェックポイントヘルパー
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``_save_state(self, filename) -> None``

  ``__getstate__()`` が収集したフィールドを pickle でバージョン管理付きで保存します。
  外部ファイルの追加書き込みが必要な場合のみオーバーライドし、その際は ``super()._save_state(filename)`` を先に呼び出してください。

- ``_load_state(self, filename, mode="resume", restore_rng=True) -> None``

  チェックポイントを読み込み、``_apply_state()`` を呼び出します。
  外部ファイルの追加読み込みが必要な場合のみオーバーライドし、``super()._load_state(...)`` を先に呼び出してください。

- ``_apply_state(self, data, mode="resume", restore_rng=True) -> None``

  基底クラスの状態（MPI検証・タイマー・パラメータ）を復元します。
  アルゴリズム固有のフィールド復元や ``"continue"`` モードの意味論を実装する場合にオーバーライドします。
  常に ``super()._apply_state(data, mode=mode, restore_rng=restore_rng)`` を先に呼び出してください。

- ``__getstate__(self) -> dict``

  MROをたどって ``_checkpoint_attrs`` に列挙されたフィールドをすべて収集し、チェックポイントのスナップショット辞書を返します。
  グローバルなRNGや外部ポリシーオブジェクトなど属性以外のデータを保存する必要がある場合のみオーバーライドします。


``Algorithm`` （サブクラス）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Algorithm`` はアルゴリズムの具体的な実装を提供します。
``AlgorithmBase`` を継承し、以下のメソッドを実装してください。

``__init__``
^^^^^^^^^^^^^

.. code-block:: python

  def __init__(self, info: odatse.Info, runner: odatse.Runner = None,
               run_mode: str = "initial"):
      super().__init__(info=info, runner=runner, run_mode=run_mode)
      # アルゴリズム固有のパラメータを info から読み取る ...

``info``、``runner``、``run_mode`` を基底クラスのコンストラクタに渡してください。
``super().__init__()`` の **後で** アルゴリズム固有のパラメータを読み取ってください。
基底クラスのコンストラクタが ``rng``、``proc_dir`` などの属性を設定するため、
サブクラスの初期化はその後に行う必要があります。

``_initialize`` （必須）
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  def _initialize(self) -> None:
      # 新規実行のための状態を構築する
      # ここで runner を使ってはいけない — 初期評価は _run() で行う
      self.istep = 0
      self.best_fx = np.inf
      ...

``mode`` が ``"init"`` で始まる場合にフレームワークから呼び出されます。
runner を使用しないでください（初期評価は ``_run()`` で行います）。

``_prepare`` （必須）
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  def _prepare(self) -> None:
      # _initialize() / _load_state() の後、メインループの前に呼ばれる
      # タイマーの初期化や出力ファイルのオープンに適しています
      self.timer["run"]["submit"] = 0.0

``_initialize()`` または ``_load_state()`` の後、``_run()`` の前に呼び出されます。

``_run`` （必須）
^^^^^^^^^^^^^^^^^

.. code-block:: python

  def _run(self) -> None:
      # チェックポイントのディスパッチ（init/resume/continue）は
      # prepare() が処理済み。メインループを直接開始する。

      # init モードの場合、ここで初期評価を行う
      if self.mode.startswith("init"):
          self.fx = self.runner.submit(self.state, (0, 0))
          ...

      # メインループ
      while self.istep < self.numsteps:
          ...
          # 目的関数の評価:
          args = (self.istep, 0)
          fx = self.runner.submit(x, args)
          ...
          self.istep += 1

          # 定期的にチェックポイントを保存:
          if self.checkpoint:
              time_now = time.time()
              if self.istep >= next_checkpoint_step or time_now >= next_checkpoint_time:
                  self._save_state(self.checkpoint_file)
                  next_checkpoint_step = self.istep + self.checkpoint_steps
                  next_checkpoint_time = time_now + self.checkpoint_interval

      if self.checkpoint:
          self._save_state(self.checkpoint_file)

アルゴリズムの本体を記述します。
チェックポイントのディスパッチは完了しているため、``_run()`` 内で ``self.mode`` を確認するのは
初回評価ステップに固有の処理（``mode.startswith("init")``）の場合のみです。

探索パラメータ ``x`` に対して目的関数の値 ``f(x)`` を得るには次のようにします。

.. code-block:: python

  args = (step, set)
  fx = self.runner.submit(x, args)

``_post`` （必須）
^^^^^^^^^^^^^^^^^^

.. code-block:: python

  def _post(self) -> dict:
      # 結果をファイルに書き出し、MPIランクから収集する、など
      return {"x": self.xopt, "fx": self.best_fx}

アルゴリズムの後処理を記述し、結果を辞書形式で返します。
``output_dir`` から呼び出されます。

チェックポイントフィールド： ``_checkpoint_attrs``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

チェックポイントで保存・復元する属性名を列挙したクラス変数 ``_checkpoint_attrs`` を宣言します。

.. code-block:: python

  class Algorithm(odatse.algorithm.AlgorithmBase):
      _checkpoint_attrs: list[str] = ["istep", "best_x", "best_fx"]

基底クラスの ``__getstate__()`` がMROをたどって ``_checkpoint_attrs`` の全フィールドを
自動的に収集します。単純な場合はオーバーライド不要です。

``"continue"`` モードの意味論など、独自の復元ロジックが必要な場合は ``_apply_state()`` をオーバーライドします。

.. code-block:: python

  def _apply_state(self, data: dict, mode: str = "resume",
                   restore_rng: bool = True) -> None:
      super()._apply_state(data, mode=mode, restore_rng=restore_rng)
      # アルゴリズム固有のフィールドを復元:
      self.istep   = data["istep"]
      self.best_x  = data["best_x"]
      self.best_fx = data["best_fx"]
      if mode == "continue":
          # スケジュールの延長やカウンタの更新など
          ...


最小構成の実装例
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  import numpy as np
  import time
  import odatse

  class Algorithm(odatse.algorithm.AlgorithmBase):
      """グリッドサーチのアルゴリズム実装例"""

      _checkpoint_attrs: list[str] = ["icount", "best_x", "best_fx", "results"]

      def __init__(self, info, runner=None, run_mode="initial"):
          super().__init__(info=info, runner=runner, run_mode=run_mode)
          self.mesh = [...]   # info から読み取る

      def _initialize(self) -> None:
          self.icount = 0
          self.best_fx = np.inf
          self.best_x = None
          self.results = []

      def _prepare(self) -> None:
          self.timer["run"]["submit"] = 0.0

      def _run(self) -> None:
          next_chk_step = self.icount + self.checkpoint_steps
          next_chk_time = time.time() + self.checkpoint_interval

          while self.icount < len(self.mesh):
              x = np.array(self.mesh[self.icount])
              args = (self.icount, 0)
              time_sta = time.perf_counter()
              fx = self.runner.submit(x, args)
              self.timer["run"]["submit"] += time.perf_counter() - time_sta

              self.results.append((x, fx))
              if fx < self.best_fx:
                  self.best_fx, self.best_x = fx, x.copy()
              self.icount += 1

              if self.checkpoint:
                  now = time.time()
                  if self.icount >= next_chk_step or now >= next_chk_time:
                      self._save_state(self.checkpoint_file)
                      next_chk_step = self.icount + self.checkpoint_steps
                      next_chk_time = now + self.checkpoint_interval

          if self.checkpoint:
              self._save_state(self.checkpoint_file)

      def _post(self) -> dict:
          if odatse.mpi.algrank() == 0:
              with open("result.txt", "w") as f:
                  f.write(f"fx = {self.best_fx}\n")
          return {"x": self.best_x, "fx": self.best_fx}


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

- ``initialize(self, rng, limitation, num_walkers)`` を呼んで初期値の設定を行います。
  引数は乱数発生器 ``rng``、制約条件 ``limitation``、walker の数 ``num_walkers`` です。


``MeshGrid`` クラス
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

離散的なパラメータ空間を定義するためのヘルパークラスです。

- コンストラクタ引数は ``Info`` または ``param=`` にdict形式のパラメータを取ります。

  - ``Info`` 型の引数の場合、 ``Info.algorithm.param`` から探索範囲の最小値・最大値・単位や初期値を取得します。
  - dict 型の引数の場合は ``Info.algorithm.param`` 相当の内容を辞書形式で受け取ります。
  - 詳細は :ref:`mapper の入力ファイル <mapper_input_param>` を参照してください。

- ``do_split(self)`` メソッドは、候補点の集合を分割して各MPIランクに配分します。
- 入出力について

  - ``from_file(cls, path)`` クラスメソッドは、 ``path`` からメッシュデータを読み込んで ``MeshGrid`` クラスのインスタンスを作成します。
  - ``store_file(self, path)`` メソッドは、メッシュの情報を ``path`` のファイルに書き出します。
