共通事項
================================

``Solver`` と ``Algorithm`` に共通する事柄について説明します。

``odatse.Info``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
入力パラメータを扱うためのクラスです。
インスタンス変数として次の4つの ``dict`` を持ちます。

- ``base``

  - ディレクトリ情報など、プログラム全体で共通するパラメータ

- ``solver``

  - ``Solver`` が用いる入力パラメータ

- ``algorithm``

  - ``Algorithm`` が用いる入力パラメータ

- ``runner``

  - ``Runner`` が用いる入力パラメータ


``Info`` は ``base``, ``solver``, ``algorithm``, ``runner`` の4つのキー(省略可)を持つ ``dict`` を渡して初期化できます。また、クラスメソッド ``from_file`` に TOML形式の入力ファイルへのパスを渡して作成することもできます。


``base`` について
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

要素として計算のルートディレクトリ ``root_dir`` と出力のルートディレクトリ ``output_dir`` が自動で設定されます。

- ルートディレクトリ ``root_dir``

  - デフォルトはカレントディレクトリ ``"."`` です
  - 絶対パスに変換されたものがあらためて ``root_dir`` に設定されます
  - 先頭の ``~`` はホームディレクトリに展開されます
  - 具体的には次のコードが実行されます

    .. code-block:: python

      p = pathlib.Path(base.get("root_dir", "."))
      base["root_dir"] = p.expanduser().absolute()

- 出力ディレクトリ ``output_dir``

  - 先頭の ``~`` はホームディレクトリに展開されます
  - 絶対パスが設定されていた場合はそのまま設定されます
  - 相対パスが設定されていた場合、 ``root_dir`` を起点とした相対パスとして解釈されます
  - デフォルトは ``"."`` 、つまり ``root_dir`` と同一ディレクトリです
  - 具体的には次のコードが実行されます

    .. code-block:: python

      p = pathlib.Path(base.get("output_dir", "."))
      p = p.expanduser()
      base["output_dir"] = base["root_dir"] / p


``odatse.Runner``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Runner`` は ``Algorithm`` と ``Solver`` とをつなげるためのクラスです。
コンストラクタ引数として ``Solver`` のインスタンス、 ``Info`` のインスタンス、パラメータの変換を記述する ``Mapping`` のインスタンス、制約条件を記述する ``Limitation`` のインスタンスを取ります。
``Mapping`` のインスタンスを省略した場合は変換しない ``TrivialMapping`` が仮定されます。
``Limitation`` のインスタンスを省略した場合は、制約を課さない ``Unlimited`` が仮定されます。

``submit(self, x: np.ndarray, args: Tuple[int,int]) -> float`` メソッドは、探索パラメータ ``x`` とオプションパラメータ ``args`` に対してソルバーを実行し、結果として目的関数の値 ``f(x)`` を返します。
関数を評価する際に ``Limitation`` のインスタンスを用いて探索パラメータ ``x`` が制約を満たしているかを確認します。次に ``Mapping`` のインスタンスを用いて ``x`` から実際にソルバーが使う入力 ``y = mapping(x)`` を得ます。

その他、 ``Runner`` で使われる ``info`` の詳細は :doc:`../input/index` を参照してください。


``odatse.Mapping``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Mapping`` は逆問題解析アルゴリズムの探索パラメータから順問題ソルバーの変数へのマッピングを記述するクラスです。 ``__call__(self, x: np.ndarray) -> np.ndarray`` メソッドを持つ関数オブジェクトとして定義します。
現在は自明な変換 ``TrivialMapping`` とアフィン写像 ``Affine`` のクラスが定義されています。

``TrivialMapping``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
自明なマッピング :math:`x\to x` (変換しない)を与えるクラスです。
Runner クラスの引数のデフォルト値となります。

``Affine``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
アフィン変換 :math:`x \to y=Ax+b` を与えるクラスです。
要素 ``A``, ``b`` はコンストラクタの引数に指定するか、 ``from_dict`` クラスメソッドに辞書の形式で与えます。
ODAT-SEの入力ファイルに指定する場合、パラメータの指定方法は :doc:`../input/runner` の ``[runner.mapping]`` セクションを参照してください。


``odatse.Limitation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Limitation`` は逆問題解析アルゴリズムで探索する :math:`N` 次元のパラメータ :math:`x` に課す制約条件を与えるクラスです。 ``judge(self, x: np.ndarray) -> bool`` のメソッドを持つクラスとして定義します。
現在は制約なし ``Unlimited`` と線形不等式制約 ``Inequality`` のクラスが定義されています。

``Unlimited``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
制約条件を課さないというクラスです。 ``judge`` メソッドは常に ``True`` を返します。
Runner クラスの引数のデフォルト値となります。

``Inequality``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:math:`N` 次元の探索パラメータ :math:`x` に対して、 :math:`M` 行 :math:`N` 列の行列 :math:`A` と :math:`M` 次元のベクトル :math:`b` で与えられる :math:`M` 個の制約条件 :math:`A x + b > 0` を課すクラスです。

要素 ``A``, ``b`` はコンストラクタの引数に指定するか、 ``from_dict`` クラスメソッドに辞書の形式で与えます。
ODAT-SEの入力ファイルに指定する場合、パラメータの指定方法は :doc:`../input/runner` の ``[runner.limitation]`` セクションを参照してください。


``odatse.initialize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``initialize(argv=None) -> (Info, str)`` は、コマンドライン引数の解釈と入力ファイルの読み込みをまとめて行う初期化関数です。
``odatse`` コマンドと同じ引数（入力ファイルのパス、 ``--init`` / ``--resume`` / ``--cont`` / ``--reset_rand`` / ``--nalg`` / ``--nsolve`` 。詳細は :doc:`../manual/command` を参照）を解釈し、 ``Info`` のインスタンスと実行モード文字列 ``run_mode`` の組を返します。内部で ``odatse.mpi.setup()`` も呼び出します。ただし ``setup()`` が既に呼ばれている場合（ ``odatse.mpi.ready()`` が ``True`` ）は既存の分割をそのまま使い、 ``--nalg`` / ``--nsolve`` が指定されていればそれが既存の分割と一致することを確認します。

- ``argv`` を省略した場合（ ``None`` ）は ``sys.argv[1:]`` が解釈されます。
  独自の引数処理を持つスクリプトに組み込む場合は、 ``argv`` に明示的にリストを渡すことで ``sys.argv`` に依存せずに初期化できます。

  .. code-block:: python

      info, run_mode = odatse.initialize(["input.toml", "--resume"])

- ``run_mode`` は ``"initial"``, ``"resume"``, ``"continue"`` のいずれか（ ``--reset_rand`` 指定時は ``"-resetrand"`` が付加）です。
  ``Algorithm`` のコンストラクタの ``run_mode`` 引数にそのまま渡すことで、リスタート機能が独自スクリプトでも有効になります。


``odatse.mpi``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MPI コミュニケータへのアクセスを提供するモジュールです。
mpi4py がインストールされていない環境や環境変数 ``ODATSE_NOMPI`` が設定された場合は、非 MPI のスタブとして動作します。
二層並列（アルゴリズム層 × ソルバーグループ）の詳細は :doc:`../tutorial/parallel_solver` を参照してください。

- ``setup(nalg=None, nsolve=None, comm=None)`` : コミュニケータを分割します。 ``Solver`` / ``Algorithm`` の構築前に呼ぶ必要があります（ ``odatse.initialize()`` を使う場合は内部で呼ばれます）。 ``comm`` は分割対象のイントラコミュニケータで、省略時は ``MPI.COMM_WORLD`` です。同じ実効的な構成（同じコミュニケータ、かつ省略値を補った後の ``nalg`` / ``nsolve`` が同じ）で再度呼び出した場合は何もせず、異なる構成で呼び出した場合は ``RuntimeError`` を送出します。コミュニケータは MPI のハンドルとして比較されます（mpi4py の ``==`` ）。同じハンドルを包む別の Python オブジェクトは同じコミュニケータ、複製（ ``Dup()`` ）は別のコミュニケータです。
- ``ready()`` : ``setup()`` が呼び出し済みかどうか（非 MPI のスタブでは常に ``True`` ）。
- ``comm()`` / ``size()`` / ``rank()`` : 全体のコミュニケータとそのサイズ・ランク。 ``setup()`` の前は ``MPI.COMM_WORLD`` を、後は ``setup()`` に渡したコミュニケータを指します。
- ``algcomm()`` / ``algsize()`` / ``algrank()`` : アルゴリズム層のコミュニケータとそのサイズ・ランク。
- ``solcomm()`` / ``solsize()`` / ``solrank()`` : ソルバーグループのコミュニケータとそのサイズ・ランク。
- ``run_on_algorithm()`` : 呼び出したプロセスがアルゴリズム層に属するかどうか。
- ``enabled()`` : MPI が利用可能かどうか（ ``ODATSE_NOMPI`` 設定時は ``False`` ）。

``algcomm()`` , ``solcomm()`` とそのサイズ・ランクのアクセサ、および ``run_on_algorithm()`` は、 ``setup()`` の前に呼ぶと ``RuntimeError`` を送出します。これらの上に構築されたものも同様で、 ``Solver`` / ``Algorithm`` のほか、メッシュファイルを読み込み・分配する ``MeshGrid`` （ ``MeshGrid(info)`` 、 ``MeshGrid.from_file()`` 、 ``store_file()`` 、 ``do_split()`` ）も ``setup()`` の後に使う必要があります。

ODAT-SE を別の MPI プログラムの中からライブラリとして使う場合、そのプログラムが既に ``setup()`` を呼んでいるかどうかは分かりません。 ``ready()`` で確認し、必要な場合にのみコミュニケータを分割してください。

.. code-block:: python

    import odatse.mpi

    if not odatse.mpi.ready():
        odatse.mpi.setup(comm=my_comm)   # my_comm: ODAT-SE が使ってよいランクの集合

コミュニケータを渡す場合の注意:

- ``setup()`` は ``comm`` 上の集団操作です。 ``comm`` の全ランクが同じ引数で呼び出す必要があります。 ``nalg * nsolve`` は ``comm`` のサイズに一致しなければなりません。
- ``setup(comm=...)`` は入力ファイルの読み込みより **前** に呼んでください。 ``Info.from_file()`` は ``comm()`` 上で入力をブロードキャストしますが、 ``setup()`` の前の ``comm()`` は ``MPI.COMM_WORLD`` なので、一部のランクだけで先に入力を読み込むとデッドロックします。その後に ``odatse.initialize()`` を呼んでも ``setup(comm=...)`` による分割はそのまま使われます（上記参照）。
- ``setup()`` を取り消す方法はありません。分割はプロセスの終了まで有効で、その後に別のコミュニケータや別のレイアウトで ``setup()`` を呼ぶと ``RuntimeError`` になります。
- ``comm`` の所有権は呼び出し側にあります。ODAT-SE は ``comm`` を解放しません。ODAT-SE の使用中は有効なまま保持してください。
- 複製（ ``comm.Dup()`` ）は別のコミュニケータとして扱われます。元のコミュニケータで ``setup()`` を呼んだ後に複製を渡すと ``RuntimeError`` になります。
- ソルバーワーカーが ``evaluate`` の外側で失敗した場合、ODAT-SE は ``comm`` に対して ``MPI_Abort`` を呼びます。MPI の実装によっては ``comm`` のランクだけでなくジョブ全体が終了します。
