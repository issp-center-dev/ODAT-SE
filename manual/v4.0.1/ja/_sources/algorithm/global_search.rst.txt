=========================================
大域最適化 ``global_search``
=========================================

.. _scipy.optimize.differential_evolution: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
.. _scipy.optimize.shgo: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html
.. _scipy.optimize.direct: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.direct.html
.. _scipy.optimize.dual_annealing: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html

``global_search`` は scipy.optimize の大域最適化ルーチンを用いて
:math:`f(x)` の最小化を行います。
現在は以下の手法が利用できます。

- 差分進化法 (differential evolution, `scipy.optimize.differential_evolution`_):
  個体群(population)を維持し、個体間の差分ベクトルから新しい候補点を生成する
  進化計算法です。微分を必要とせず、多峰性の問題に対してロバストです。
- shgo (simplicial homology global optimization, `scipy.optimize.shgo`_):
  サンプリング点から単体複体を構成し、その位相構造に基づいて局所最適化の
  開始点を系統的に選ぶ決定論的手法です。発見した **すべての局所解のリスト** を
  出力できるのが特徴です。
- direct (DIviding RECTangles, `scipy.optimize.direct`_):
  探索領域を超矩形に分割し、有望かつ大きい矩形を優先的に細分化していく
  決定論的手法です。乱数を使用せず、完全な再現性があります。
- dual annealing (`scipy.optimize.dual_annealing`_):
  一般化シミュレーテッドアニーリング (GSA) に基づく確率的手法です。
  焼きなましによる大域探索に、受理された点からの局所最適化を組み合わせます。

探索範囲は ``[algorithm.param]`` の ``min_list`` / ``max_list`` で規定され、
scipy の ``bounds`` 引数として渡されます。初期値 (``initial_list``) は使用しません。

MPI 並列
~~~~~~~~~~~~~~~~~

MPI 実行時には、アルゴリズムランク 0 が最適化ルーチンを駆動し、
他のランクは評価サーバーとして動作します。差分進化法では 1 世代分の候補点が、
shgo ではサンプリング段階の評価点が、まとめて各ランクに分配され、
各ランクは自身のソルバーグループで評価を行います。
点レベルの並列度(アルゴリズムランク数)とソルバー内並列度 (``nsolve``) を
組み合わせた 2 階層の並列化が可能です。

差分進化法の 1 世代あたりの目的関数の評価回数は ``popsize`` × 次元数であり、
進化段階の総評価回数はおおよそ (``maxiter`` + 1) × ``popsize`` × 次元数が上限になります
(収束判定により早く終了する場合があります)。
この見積もりは既定の初期集団の作り方を前提としており、``init`` で初期集団を明示的に与えた場合や、
上限と下限が等しい次元がある場合には当てはまりません。
また、既定で有効な最終段階の精錬 (``polish``) による評価回数は含みません。
shgo の局所精錬(内部の局所最適化)はランク 0 上で逐次実行されます。

direct と dual annealing は並列評価に対応していないため、ランク 0 上で
逐次実行されます
(他のランクは待機します。各点内のソルバー並列 ``nsolve`` は有効です)。

前準備
~~~~~~

あらかじめ `scipy <https://docs.scipy.org/doc/scipy/reference>`_ を
インストールしておく必要があります。 ::

  python3 -m pip install scipy

入力パラメータ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

サブセクション ``param`` と ``global_search`` を持ちます。

``[algorithm.param]`` セクション
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``min_list``

  形式: 実数のリスト。長さはdimensionの値と一致させます。

  説明: パラメータが取りうる最小値。

- ``max_list``

  形式: 実数のリスト。長さはdimensionの値と一致させます。

  説明: パラメータが取りうる最大値。

- ``unit_list``

  形式: 実数のリスト。長さはdimensionの値と一致させます。

  説明: 各パラメータの単位。探索アルゴリズム中では、各パラメータをそれぞれこれらの値で割ることで、簡易的な無次元化・正規化を行います。定義しなかった場合にはすべての次元で 1.0 となります。

``[algorithm.global_search]`` セクション
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最適化手法とそのハイパーパラメータを設定します。

``method`` 以外のパラメータは、そのまま選択した scipy 関数の引数として渡されます。
受け付けられないパラメータ名が指定された場合は、最適化を開始する前に
エラーで終了します。``bounds``, ``workers``, ``seed`` など ODAT-SE が管理する
引数は指定できません。

- ``method``

  形式: 文字列 (default: "DE")

  説明: 最適化手法の名前(大文字小文字は区別しません)。
  "DE" または "differential_evolution" で差分進化法、"shgo" で shgo、
  "direct" で direct、"dual_annealing" で dual annealing を選択します。

- その他のパラメータ

  選択した scipy 関数の引数をそのまま指定できます。
  詳細は scipy のドキュメントを参照してください。

  - 差分進化法: ``popsize``, ``maxiter``, ``tol``, ``mutation``,
    ``recombination``, ``strategy``, ``polish`` など。
    乱数は ``[algorithm]`` セクションの ``seed`` から初期化されます
    (アルゴリズムランク 0 の乱数系列が使用されます)。
  - shgo: ``n``, ``iters``, ``sampling_method`` など。
    サブテーブル ``[algorithm.global_search.options]`` および
    ``[algorithm.global_search.minimizer_kwargs]`` はそれぞれ scipy の
    ``options`` / ``minimizer_kwargs`` 引数として渡されます。
    shgo は決定論的で乱数を使用しません。
  - direct: ``maxfun``, ``maxiter``, ``eps``, ``locally_biased``,
    ``len_tol``, ``vol_tol`` など。direct も決定論的で乱数を使用しません。
  - dual annealing: ``maxiter``, ``maxfun``, ``initial_temp``,
    ``restart_temp_ratio``, ``visit``, ``accept``, ``no_local_search``,
    ``x0`` など。サブテーブル
    ``[algorithm.global_search.minimizer_kwargs]`` は局所最適化に渡す
    ``minimizer_kwargs`` 引数となります (scipy >= 1.8。
    それより古いバージョンではこの引数は ``local_search_options``
    という名前です)。乱数は ``[algorithm]`` セクションの
    ``seed`` から初期化されます。

設定例:

.. code-block:: toml

    [algorithm]
    name = "global_search"
    seed = 12345

    [algorithm.param]
    min_list = [-5.0, -5.0]
    max_list = [ 5.0,  5.0]

    [algorithm.global_search]
    method = "DE"
    popsize = 15
    maxiter = 100

注意点
~~~~~~~~~~~~~~~~~

- 差分進化法で ``polish`` (default: true) が有効な場合、終了後に L-BFGS-B 法による
  局所最適化が実行されます。この局所最適化はランク 0 上で逐次実行され、
  勾配は数値差分により評価されます(勾配 1 回あたり次元数+1 回のソルバー実行)。
  ソルバーの評価コストが大きい場合は ``polish = false`` も検討してください。
- shgo の局所精錬もランク 0 上で逐次実行されます。デフォルトの局所最適化手法は
  SLSQP で、勾配は数値差分により評価されます。
- shgo の並列評価 (``workers``) は scipy >= 1.11 が必要です
  (シリアル実行ではこの制限はありません)。
  それ未満のバージョンで MPI 並列実行した場合は開始前にエラーで停止します。
- direct は scipy >= 1.9 が必要です。また、最適点近傍の精密化が遅いため、
  direct で当たりをつけてから minsearch で磨く使い方が有効です。
- dual annealing はデフォルトで焼きなまし中に局所最適化 (L-BFGS-B 法) を
  実行します。勾配は数値差分により評価されるため、ソルバーの評価コストが
  大きい場合は評価回数が増大します。``no_local_search = true`` とすると
  局所最適化を行わない古典的な焼きなましになります。
  ``maxfun`` (デフォルト: 1e7) で総評価回数を制限できます。
- ``[runner.limitation]`` による制約条件は、制約を満たさない点の目的関数値を
  無限大とみなす方法で処理されます。
- リスタート(チェックポイント)には対応していません(後述の「リスタート」の節を参照)。

出力ファイル
~~~~~~~~~~~~~~~~~

``GenerationData.txt`` / ``IterationData.txt`` / ``MinimumData.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

反復ごとの最良点の情報を出力します(ランク 0 のみ)。
差分進化法では ``GenerationData.txt`` に、世代番号、最良点の変数の値、
目的関数の値、収束度 (convergence) がこの順に出力されます。
shgo / direct では ``IterationData.txt`` に、反復番号、最良点の変数の値、
目的関数の値がこの順に出力されます。
dual annealing では ``MinimumData.txt`` に、より良い最小値が見つかるたびに、
番号、変数の値、目的関数の値、context (0: 焼きなまし中に発見、
1: 局所最適化中に発見、2: dual annealing 過程で発見) がこの順に出力されます。

``LocalMinimaData.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

shgo の場合のみ出力されます(ランク 0 のみ)。
発見されたすべての局所解について、番号、変数の値、目的関数の値を
目的関数の値の昇順で出力します。

``History_FunctionCall.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

各ランクが評価した目的関数の呼び出し履歴を、ランクごとの出力ディレクトリに
記録します。各行には、呼び出し番号、変数の値、目的関数の値がこの順に出力されます。

``res.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最終的に得られた目的関数の値とその時のパラメータの値を記載しています。

.. code-block::

    fx = 4.119494492750836e-11
    x1 = 6.135735138280041e-06
    x2 = 1.4614801832975428e-06

リスタート
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``global_search`` による探索はリスタートに対応していません。
