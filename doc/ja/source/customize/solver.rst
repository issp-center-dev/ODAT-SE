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

    .. note::
      ``--nsolve`` が 2 以上の場合、 ``evaluate`` や ``gradient`` を呼ぶときのカレントディレクトリは、
      コントローラでは ``proc_dir`` ですが、ワーカーでは ``odatse`` を起動したディレクトリです。
      ファイルを読み書きする場合は、 ``self.work_dir`` を明示的に使ってください。

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
    ``Runner`` から呼ばれる場合のエラーの扱い:

    ``evaluate`` が ``RuntimeError`` を送出した場合、 ``[runner]`` セクションで ``ignore_error = true`` が指定されていると、Runner は例外を無視して目的関数値を ``np.nan`` として扱います。
    また、探索点が制約条件 (``[runner.limitation]``) を満たさない場合はソルバーは呼び出されず、目的関数値は ``np.inf`` になります。

  .. note::
    ソルバー並列を使用する場合 (``--nsolve`` が 2 以上)、 ``evaluate`` はソルバーグループ内の全 MPI ランクで同一の ``x``, ``args`` を引数として呼び出されます。
    ランク間の役割分担はソルバー内で実装してください。詳細は :doc:`../tutorial/parallel_solver` を参照してください。
    ソルバーグループ内のどのランクで ``evaluate`` が例外を送出しても、コントローラランク (``solrank() == 0``) に報告されてそこで処理されます。失敗した全ランクの例外が ``RuntimeError`` であれば ``ignore_error`` が適用され、それ以外の場合はジョブはエラー終了します。ランクが失敗した場合でも、 ``evaluate`` 内の集団通信はグループ全体で揃っている必要があります。

- 勾配への対応 (任意)

  目的関数の勾配が計算できる場合、次のメンバーを実装すると、 ``Algorithm`` が勾配を利用できるようになります (例: ``minsearch`` での ``use_gradient = true``)。

  - ``supports_gradient`` (クラス属性、デフォルト: False)

    勾配を計算できるソルバーでは True にします。派生クラスでは、クラス属性
    (``supports_gradient = True``)、インスタンス属性
    (``__init__`` 内の ``self.supports_gradient = True``)、True を返すプロパティの
    いずれでも宣言できます。

  - ``gradient(self, x, args=()) -> np.ndarray``

    入力 ``x`` における目的関数の勾配を、 ``x`` と同じ shape で返します。

    ``evaluate`` メソッドと同様の引数をとります。
    ``function.Solver`` では、このメソッドを実装する代わりに ``set_gradient()`` を使えます (:doc:`function_solver` を参照)。

    .. note::
      ``Runner`` から呼ばれる場合のエラーの扱い:

      ``gradient`` が ``RuntimeError`` を送出した場合、 ``[runner]`` セクションで ``ignore_error = true`` が指定されていると、Runner は例外を無視して勾配を ``np.nan`` の配列として扱います。
      また、探索点が制約条件 (``[runner.limitation]``) を満たさない場合はソルバーは呼び出されず、勾配の値は ``np.nan`` の配列になります。

    ``gradient`` は、 ``evaluate`` の返り値を、 ``evaluate`` に渡されるのと同じ ``x`` で微分した値を返す必要があります。
    その正しさはソルバーの実装者の責任です。ODAT-SE は返り値の shape と失敗の有無しか確かめないので、
    符号や倍率を誤った勾配は、最適化の早すぎる終了や誤った点への収束として現れ、出力からは見分けられません。
    次の ``check_gradient`` を使って、あらかじめ差分と見比べてください。

  - ``check_gradient(self, x, args=(0, 0), step=None) -> tuple[np.ndarray, np.ndarray]``

    ``SolverBase`` が提供する確認用のメソッドです (派生クラスで実装する必要はありません)。
    点 ``x`` での ``gradient`` の値 ``g`` と、 ``evaluate`` の中心差分で求めた勾配 ``d`` を、組 ``(g, d)`` として返します。
    合否は判定せず、何も表示しません。許容誤差は呼び出す側で決めてください。

    - ``x``: 1 次元の有限な実数の並び。 ``evaluate`` に渡すのと同じ空間の値です。
    - ``args``: ``evaluate`` と ``gradient`` のすべての呼び出しにそのまま渡されます。
    - ``step``: 差分の刻み。正の有限な値で、 ``x`` に対する相対値ではなく、座標に直接足し引きする大きさです。スカラー、または ``x`` と同じ長さの並び。
      省略した場合は成分ごとに ``eps**(1/3) * max(1, |x_i|)`` (``eps`` は倍精度の機械イプシロン) を使います。
      これは滑らかな関数での目安です。評価値にノイズがある場合や、変数の尺度が 1 から大きく外れている場合は指定してください。
    - ``x`` や ``step`` が不正な場合は、ソルバーを呼ぶ前に ``ValueError`` を送出します。
      ``evaluate`` や ``gradient`` が送出した例外はそのまま伝わります。

    ``evaluate(x)`` 、 ``gradient(x)`` 、続いて成分ごとに ``evaluate(x + h e_i)`` 、 ``evaluate(x - h e_i)`` の順に呼びます。
    ``evaluate`` の呼び出しは 2 × 次元 + 1 回、 ``gradient`` は 1 回です。
    ``g`` の shape は ``gradient`` が返したままで、検査されません。 ``d`` の shape は ``x`` と同じです。
    刻みが小さすぎて ``x + h`` と ``x - h`` が同じ値になる成分は評価されず、 ``d`` の値は ``np.nan`` になります。

    ソルバーのテストでは、たとえば次のように使います。

    .. code-block:: python

      import numpy as np

      def test_gradient(solver):
          solver.work_dir.mkdir(parents=True, exist_ok=True)
          x = np.array([0.3, -1.2])
          g, d = solver.check_gradient(x)
          assert g.shape == d.shape
          np.testing.assert_allclose(g, d, rtol=1e-6, atol=1e-8)

    .. note::
      - ``Runner`` を経由しないので、作業ディレクトリの作成など、 ``evaluate`` に必要な準備は呼び出す側で済ませてください。
        探索範囲や制約条件も考慮されません。範囲内に収まる点と刻みを選んでください。
      - 呼び出しの後、ソルバーの内部状態や作業ディレクトリのファイルは、最後の差分用の評価のものになります。
      - ``np.testing.assert_allclose`` はスカラーと配列の比較を通すので、shape は別に確かめてください。
      - 差分との一致が意味を持つのは、その点で目的関数が微分可能で、評価が再現する場合だけです
        (たとえば Ackley 関数は原点で微分できません)。刻みに迷ったら、複数の ``step`` で見比べてください。
      - 確かめられるのは指定した点での値だけです。複数の点で確かめることを勧めます。
      - ソルバー並列では、グループの全ランクが同じ引数で呼び出してください。例外はランク間で集約されません。


  ``runner.log`` が数えるのは ``evaluate()`` の呼び出しだけです。勾配の呼び出しは記録されず、回数にも含まれません。

  ``[runner.mapping]`` を指定した場合は、勾配の利用に対応していません。
