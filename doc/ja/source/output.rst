出力ファイル
=====================

各種 ``Solver``, ``Algorithm`` が出力するファイルについては、 :doc:`solver/index` および :doc:`algorithm/index` を参照してください。


アルゴリズム別出力ファイル早見表
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

各アルゴリズムが出力する主なファイルの一覧です。
``RANK/`` は MPI ランクごとのサブフォルダ（``output_dir/0/`` など）を表します。
``#`` は温度点のインデックスです。各ファイルの列の意味などの詳細は、各アルゴリズムのページの「出力ファイル」の節を参照してください。

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - アルゴリズム
     - 主な出力ファイル
     - 内容
   * - :doc:`minsearch <algorithm/minsearch>`
     - ``res.txt``, ``RANK/SimplexData.txt``, ``RANK/History_FunctionCall.txt``, ``RANK/BasinHoppingData.txt``
     - 最適化結果、シンプレックスの探索経路、関数評価の履歴、ベイスンホッピングの各ホップの結果（``basinhopping`` 有効時のみ）
   * - :doc:`global_search <algorithm/global_search>`
     - ``res.txt``, ``RANK/History_FunctionCall.txt``, ``0/GenerationData.txt`` または ``0/IterationData.txt``, ``0/LocalMinimaData.txt``
     - 最適化結果、関数評価の履歴（ランクごと）、反復ごとの最良点（差分進化法では ``GenerationData.txt``、shgo・direct では ``IterationData.txt``。ランク 0 のみ）、発見された局所解の一覧（shgo の場合のみ、ランク 0 のみ）
   * - :doc:`mapper <algorithm/mapper_mpi>`
     - ``ColorMap.txt``
     - 各格子点の座標と目的関数値（ファイル名は ``colormap`` で変更可能）
   * - :doc:`random_search <algorithm/random_search>`
     - ``ColorMap.txt``
     - 各サンプル点の座標と目的関数値
   * - :doc:`bayes <algorithm/bayes>`
     - ``BayesData.txt``
     - 各ステップの推定最適値と評価点の履歴
   * - :doc:`ttopt <algorithm/ttopt>`
     - ``res.txt``, ``ttopt_hyperparameters.txt``, ``ttopt_history.txt``, ``ttopt_eval_history.txt``
     - 最適化結果、ハイパーパラメータ、最良値の更新履歴、評価履歴（オプション）
   * - :doc:`exchange <algorithm/exchange>`
     - ``RANK/trial.txt``, ``RANK/result.txt``, ``result_T#.txt``, ``best_result.txt``, ``fx.txt``
     - 提案・採択されたサンプル、温度別ログ、最良解、温度ごとの統計量
   * - :doc:`pamc <algorithm/pamc>`
     - ``RANK/trial_T#.txt``, ``RANK/trial.txt``, ``RANK/result_T#.txt``, ``RANK/result.txt``, ``RANK/weight.txt``, ``best_result.txt``, ``fx.txt``, ``pr.txt``
     - 提案・採択されたサンプル（温度別/全体）、レプリカの重み、最良解、温度ごとの統計量、分配関数比

このほかに、アルゴリズムに依らない共通ファイルとして ``time.log`` が出力されます。
また、設定に応じて ``runner.log`` （``runner.log.interval`` が正の整数の場合）および ``status.pickle`` （チェックポイント機能に対応したアルゴリズムでチェックポイント機能が有効な場合）も出力されます。詳細は以下のとおりです。


共通ファイル
~~~~~~~~~~~~~~~~~~

``time.log``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
計算にかかった総時間を出力します。
アルゴリズム側のランク 0 のプロセスのみが書き出すため、出力先は ``output_dir/0/time.log`` です。
計算の初期化にかかった時間、計算の前処理にかかった時間、計算にかかった時間、計算の後処理にかかった時間について、
``init``, ``prepare``, ``run``, ``post`` のセクションごとに記載されます。

以下、出力例です。

.. code-block::

    #in units of seconds
    #init
     total = 0.4090206250548363
    #prepare
     total = 0.0002522082068026066
    #run
     total = 0.017200791044160724
     - min_search = 0.016241166973486543
    #post
     total = 0.0016664580907672644

``init`` はアルゴリズム開始前の処理（入力ファイルの解析、ソルバーおよびアルゴリズムの構築）、``prepare`` はアルゴリズムの準備処理、``run`` は主計算処理、``post`` は後処理にかかった時間を示します。
``run`` セクション内の項目はアルゴリズムや実行環境、設定によって異なります。


``runner.log``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ソルバー呼び出しに関するログ情報を出力します。
アルゴリズム側のランクごとに、そのサブフォルダ以下に出力されます
（ソルバー並列を用いる場合、ソルバー側のワーカープロセスは出力しません）。
入力で ``runner.log.interval`` パラメータが正の整数の時のみ出力されます。
ソルバー呼び出しは **すべて** 記録され、この値は何件バッファに溜まった時点でファイルへ書き出すかを指定します。
例えば ``runner.log.interval = 10`` の場合、10 件ごとにまとめて書き出されます（10 回に 1 回だけ記録されるわけではありません）。

ログの各列は以下の情報を表しています：

- 1列目：ソルバー呼び出しの通し番号
- 2列目：前回呼び出しからの経過時間（秒）
- 3列目：計算開始からの経過時間（秒）

以下、出力例です。

.. code-block::

    # $1: num_calls
    # $2: elapsed time from last call
    # $3: elapsed time from start

    1      0.000844 0.000844
    2      0.000237 0.001082
    3      0.000096 0.001177
    4      0.000106 0.001283
    5      0.000119 0.001402
    6      0.000107 0.001509
       ...

なお ``runner.log.write_result`` および ``runner.log.write_input`` を有効にすると、
4列目以降に目的関数の値と入力パラメータが追加されます。


``status.pickle``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``algorithm.checkpoint`` が true の場合、実行中の状態を ``status.pickle``
（``algorithm.checkpoint_file`` パラメータを指定した場合はそのファイル名）に出力します。
アルゴリズム側のランクごとに、そのサブフォルダ以下に出力されます
（ソルバー並列を用いる場合、ソルバー側のワーカープロセスは出力しません）。
実行を再開する場合に読み込まれます。ファイルの内容はアルゴリズムに依ります。

チェックポイント機能に対応しているのは ``exchange``, ``pamc``, ``mapper``,
``random_search``, ``bayes``, ``ttopt`` です。
``minsearch`` と ``global_search`` は ``algorithm.checkpoint`` を true にしても
チェックポイントを出力しません。

チェックポイント機能を使用すると、長時間の計算が途中で中断された場合でも、最後に保存された状態から計算を再開することができます。
再開するには、同じ入力ファイルと同じ MPI プロセス構成を用いて ``odatse --resume input.toml`` を実行します。
前回の結果を引き継いだまま続きを計算する場合は ``odatse --cont input.toml`` を使用します。
再開時または継続時に乱数系列を変更したい場合は、 ``--reset_rand`` オプションを併用します。
