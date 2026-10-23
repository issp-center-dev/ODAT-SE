======================================================
局所最適化アルゴリズムによる最適値探索 ``minsearch``
======================================================

.. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
.. _scipy.optimize.basinhopping: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html

``minsearch`` は局所最適化アルゴリズムによって最適値探索を行います。
実装には SciPy の `scipy.optimize.minimize`_ 関数を用いています。
最適化手法は ``[algorithm.minimize]`` セクションの ``method`` パラメータで選択します。
デフォルトは `Nelder-Mead 法 <https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method>`_
(downhill simplex 法とも呼ばれます) で、 `scipy.optimize.minimize`_ が受け付ける
その他の手法 (Powell, COBYLA など) も選択できます。
Nelder-Mead 法では、 パラメータ空間の次元を :math:`D` として、 :math:`D+1` 個の座標点の組を、各点での目的関数の値に応じて系統的に動かすことで最適解を探索します。

重要なハイパーパラメータとして、座標の初期値があります。
これらの局所最適化手法には局所最適解にトラップされるという問題があるので、
初期値を変えた計算を何回か繰り返して結果を確認するか、オプションの
ベイスンホッピング法 (`scipy.optimize.basinhopping`_) による大域最適化
(ランダムなホップと ``method`` で選んだ手法による局所最適化を繰り返す方法。
後述の ``basinhopping`` パラメータを参照) の利用をおすすめします。


前準備
~~~~~~

あらかじめ `scipy <https://docs.scipy.org/doc/scipy/reference>`_ をインストールしておく必要があります。 ::

  python3 -m pip install scipy

入力パラメータ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

サブセクション ``param`` と ``minimize`` を持ちます。

.. _minsearch_input_param:

``[algorithm.param]`` セクション
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``initial_list``

  形式: 実数のリスト。長さはdimensionの値と一致させます。

  説明: パラメータの初期値。 定義しなかった場合は一様ランダムに初期化されます。

- ``unit_list``

  形式: 実数のリスト。長さはdimensionの値と一致させます。

  説明: 各パラメータの単位。探索アルゴリズム中では、各パラメータをそれぞれこれらの値で割ることで、簡易的な無次元化・正規化を行います。定義しなかった場合にはすべての次元で 1.0 となります。

- ``min_list``

  形式: 実数のリスト。長さはdimensionの値と一致させます。

  説明: パラメータが取りうる最小値。最適化中にこの値を下回るパラメータが出現した場合、ソルバーは評価されずに、値が無限大だとみなされます。

- ``max_list``

  形式: 実数のリスト。長さはdimensionの値と一致させます。

  説明: パラメータが取りうる最大値。最適化中にこの値を上回るパラメータが出現した場合、ソルバーは評価されずに、値が無限大だとみなされます。

``[algorithm.minimize]`` セクション
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最適化手法とそのハイパーパラメータを設定します。
詳細は `scipy.optimize.minimize`_ のドキュメントを参照してください。

ODAT-SE 固有のキーである ``method``, ``initial_scale_list``,
``use_gradient``, ``basinhopping``(後述)以外のパラメータは、そのまま
`scipy.optimize.minimize`_ の ``options`` 引数に渡されます。
選択した手法が受け付けないパラメータ名が指定された場合は、
最適化を開始する前にエラーで終了します。
以下に挙げる ``xatol``, ``fatol``, ``maxiter``, ``maxfev`` のデフォルト値は
``method`` が "Nelder-Mead" の場合にのみ適用されます。

旧形式の callback は、評価済みの点では関数値を再利用し、未評価の点では
``iset = 1`` で評価します。callback の形式は scipy のバージョンと手法によります
(scipy < 1.11 では全手法、scipy 1.18 では TNC などが旧形式です)。
``(step, iset)`` ごとに作業ディレクトリを分けるソルバーでは、評価済みの点について
反復ごとの ``iset = 1`` の出力は作られなくなります。

最適化の終了後、scipy の ``success`` と ``message`` を1行で表示します。
``success`` が偽の場合は警告を付けます。ベイスンホッピングでも、取得できる範囲で
結果の状態を表示します。表示するのは、後述の ``use_gradient`` による置き換え前の
scipy の判定です。

- ``method``

  形式: 文字列 (default: "Nelder-Mead")

  説明: 最適化手法の名前。 `scipy.optimize.minimize`_ の ``method`` 引数にそのまま渡されます。
  例: "Nelder-Mead", "Powell", "COBYLA" など。
  勾配を必要とする手法 (BFGS, CG など) では、勾配が数値差分で評価されるため
  1回の勾配評価あたり次元数+1回のソルバー実行が発生することに注意してください
  (ソルバーが提供する勾配を使う場合を除きます。 ``use_gradient`` を参照)。
  また、探索範囲 (``min_list`` / ``max_list``) は、bounds に対応した手法
  (Powell, L-BFGS-B, TNC, SLSQP, trust-constr, COBYLA, COBYQA) では
  scipy の ``bounds`` 引数として渡されます。
  Nelder-Mead 法では従来通り、範囲外の点で目的関数値を無限大とみなす方法で処理されます。
  最適点が境界上にある可能性がある場合は、bounds に対応した
  L-BFGS-B, TNC, SLSQP, trust-constr などの手法をおすすめします。
  BFGS や CG など bounds に対応していない手法では、範囲外の目的関数値を無限大とみなすため、
  境界に達する手前で探索が止まる場合があります。

- ``initial_scale_list``

  形式: 実数のリスト。長さはdimensionの値と一致させます。

  説明: Nelder-Mead 法の初期 simplex を作るために、初期値からずらす差分。
  ``initial_list`` と、 ``initial_list`` に ``initial_scale_list`` の成分ひとつを足してできる dimension 個の点を合わせたものが ``initial_simplex`` として使われます。
  定義しなかった場合、各次元に 0.25 が設定されます。
  ``method`` が "Nelder-Mead" の場合のみ使用されます。

- ``use_gradient``

  形式: 真偽値 (default: false)

  説明: true のとき、数値差分の代わりにソルバーが計算した勾配を
  `scipy.optimize.minimize`_ の ``jac`` 引数として渡します (``basinhopping`` の局所最適化でも同様です)。
  カスタムソルバーで勾配を利用するには、 ``supports_gradient`` を True にし、
  ``gradient`` メソッドを実装してください (:ref:`Solver の定義 <customize_solver>` を参照)。

  真偽値以外 (文字列の ``"false"`` や整数の ``0`` など) を指定すると、
  値が真と評価されるか偽と評価されるかにかかわらず、 ``InputValueError`` になります。
  ``use_gradient = true`` の場合は、次の設定でもソルバーを評価する前に ``InputValueError`` になります。

  - 勾配を使わない手法: Nelder-Mead, Powell, COBYLA, COBYQA
    (大文字小文字は区別しません。 ``method`` を省略すると Nelder-Mead になります)。
  - ヘッセ行列を必要とする手法: dogleg, trust-ncg, trust-krylov, trust-exact
    (大文字小文字は区別しません)。ODAT-SE はヘッセ行列を渡しません。Newton-CG は利用できます。
  - 勾配に対応していないソルバー (``supports_gradient`` が False)。
  - ``[runner.mapping]`` の指定。
  - ``--nsolve`` が 2 以上のソルバー並列。

  ``InputValueError`` は ``InputError`` と ``ValueError`` の両方を継承します。
  ``odatse`` コマンドでは、これらの入力エラーを traceback なしで ``ERROR:`` の行に表示します。
  ``mpirun`` で実行した場合は、各ランクが1行ずつ報告します。
  false または省略時はソルバーの勾配を使わず、これらの制限は適用されません。

  探索範囲外や制約に違反する点では、ソルバーの ``gradient()`` を呼ばず、全要素が NaN の勾配を返します。
  ``gradient()`` の戻り値に非有限値 (NaN または無限大) が含まれる場合も失敗として扱い、
  Runner は ``SolverRuntimeError`` を送出します。ただし、 ``[runner]`` で
  ``ignore_error = true`` を指定した場合は、全要素が NaN の勾配を返します。
  ``gradient()`` が ``RuntimeError`` を送出した場合も、 ``ignore_error = true`` なら
  全要素が NaN の勾配として扱います。 ``NotImplementedError`` は常にそのまま送出されます。

  勾配を利用し、この実行中に探索範囲と制約を満たす点で非有限の勾配が生じた場合に限り、
  scipy の最終結果と、それまでに範囲・制約内で評価した有限の目的関数値の最小値を比較します。
  初期点も、範囲と制約を満たす場合だけ候補に含めます。
  scipy の最終値が非有限、または候補の最小値より大きければ、結果をその最良の評価点に置き換え、
  警告を表示します。置き換えた点での収束を scipy が認定したわけではありません。
  勾配の失敗がない場合と ``use_gradient = false`` の場合は置き換えません。

  callback は有限の勾配のキャッシュを再利用します。勾配の評価に失敗し、まだ回復していない点では、
  ``gradient()`` を再び呼ばずに NaN を記録します。scipy から勾配を要求された場合は再評価し、
  回復した後の callback ではキャッシュした有限の勾配を記録します。
  勾配評価にも関数評価と共有する step 番号を割り当てますが、勾配評価の step 番号は
  出力ファイルには記録しません。詳しくは後述の履歴ファイルの説明を参照してください。

  設定例:

  .. code-block:: toml

      [algorithm.minimize]
      method = "L-BFGS-B"
      use_gradient = true

- ``xatol``

  形式: 実数 (default: 1e-4)

  説明: Nelder-Mead 法の収束判定に使うパラメータ

- ``fatol``

  形式: 実数 (default: 1e-4)

  説明: Nelder-Mead 法の収束判定に使うパラメータ

- ``maxiter``

  形式: 整数 (default: 10000)

  説明: Nelder-Mead 法の反復回数の最大値

- ``maxfev``

  形式: 整数 (default: 100000)

  説明: 目的関数を評価する回数の最大値

- ``basinhopping``

  形式: 真偽値 または テーブル (default: false)

  説明: `scipy.optimize.basinhopping`_ による大域最適化(ベイスンホッピング法)を有効にします。
  ``method`` で指定した手法が各ホップの局所最適化に使われます。
  ``basinhopping = true`` とした場合は scipy のデフォルトパラメータで実行されます。
  サブテーブル ``[algorithm.minimize.basinhopping]`` を定義した場合も有効化され、
  その中のパラメータ (``niter``, ``stepsize``, ``T`` など) はそのまま
  `scipy.optimize.basinhopping`_ の引数として渡されます。
  受け付けられないパラメータ名が指定された場合は、最適化を開始する前にエラーで終了します。
  ``take_step``, ``seed`` など ODAT-SE が管理する引数は指定できません。

  ランダムなホップは探索範囲 (``min_list`` / ``max_list``) の内側に収まるように
  クリップされます。乱数は ``[algorithm]`` セクションの ``seed`` から初期化されます。
  局所最適化が合計 ``niter`` + 1 回実行されるため、ソルバーの総評価回数は
  おおよそ (``niter`` + 1) × (局所最適化 1 回あたりの評価回数) となる点に注意してください。
  有効時には ``initial_scale_list`` (初期 simplex) は使用されません。

  設定例:

  .. code-block:: toml

      [algorithm.minimize]
      method = "Nelder-Mead"

      [algorithm.minimize.basinhopping]
      niter = 50
      stepsize = 0.5


出力ファイル
~~~~~~~~~~~~~~~~~

``SimplexData.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最小値を求める途中経過に関する情報を出力します。
1行目はヘッダー、2行目以降に step、入力ファイルの ``[algorithm]`` セクションの ``label_list`` で定義された変数(省略時は ``x1``, ``x2``, ...)の値、最後に関数の値が出力されます。

以下、出力例です。

.. code-block::

    #step z1 z2 z3 R-factor
    0 5.25 4.25 3.5 0.015199251773721183
    1 5.25 4.25 3.5 0.015199251773721183
    2 5.229166666666666 4.3125 3.645833333333333 0.013702918021532375
    3 5.225694444444445 4.40625 3.5451388888888884 0.012635279378225261
    4 5.179976851851851 4.348958333333334 3.5943287037037033 0.006001660077530159
    5 5.179976851851851 4.348958333333334 3.5943287037037033 0.006001660077530159

.. note::
   ヘッダーの ``R-factor`` は目的関数の値の列を表します（旧 2DMAT に由来する名称です）。

``History_FunctionCall.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最適化の途中で目的関数が呼び出されるたびに、その情報を記録します。
各行には、step 番号、変数の値、目的関数の値がこの順に出力されます。
第1列の step 番号は、関数評価と勾配評価で共有します。
勾配評価でも番号が進むため、勾配を利用する場合は欠番が生じることがあります。
この番号は各評価を識別するためのもので、連番は保証されません。
このファイルの変数の値はソルバー単位 (``x / unit_list``) です。
一方、 ``History_Gradient.txt`` の変数と勾配はアルゴリズム単位です。
どちらのファイルにも、勾配評価の step 番号は記録されません。


``BasinHoppingData.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``basinhopping`` が有効な場合のみ出力されます。
局所最適化 1 回ごと(初期点からの 1 回 + ``niter`` 回のホップ)に、
ホップ番号、局所最適化で得られた変数の値、目的関数の値、
そのホップが受理されたかどうか (1/0) をこの順に出力します。

``History_Gradient.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

勾配を利用した場合 (``use_gradient = true`` かつソルバーが勾配を提供する場合) のみ出力されます。
1行目はヘッダー (``#No``、変数名、 ``fx``、各変数に対する勾配 ``grad_変数名``) です。
2行目以降には、通し番号、変数の値、目的関数の値、勾配の値がこの順に出力されます。
最初の行は初期点、それ以降は局所最適化の各反復に対応します。
変数と勾配はいずれもアルゴリズム単位です。ソルバーの目的関数を ``f`` とすると、
記録するのは ``F(x) = f(x / unit_list)`` の勾配、すなわち
``gradient(x / unit_list) / unit_list`` です。
一方、 ``History_FunctionCall.txt`` の変数の値はソルバー単位 (``x / unit_list``) です。
このファイルの通し番号は履歴の行番号であり、評価の step 番号ではありません。
勾配評価の step 番号は、出力ファイルには記録されません。

``res.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

最終的に得られた目的関数の値とその時のパラメータの値を記載しています。
最初に目的関数、その後は入力ファイルの ``[algorithm]`` セクションにある ``label_list`` で定義された変数(省略時は ``x1``, ``x2``, ...)の値が順に記載されます。

以下、出力例です。

.. code-block::

    fx = 7.382680568652868e-06
    z1 = 5.230524973874179
    z2 = 4.370622919269477
    z3 = 3.5961444501081647


リスタート
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``minsearch`` アルゴリズムはリスタートに対応していません
(選択した手法や basinhopping の有無によりません)。
