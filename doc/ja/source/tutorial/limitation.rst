制約式を適用したレプリカ交換モンテカルロ法による探索
================================================================

ここでは、 ``[runner.limitation]`` セクションに設定できる制約式機能のチュートリアルを示します。
例として、レプリカ交換モンテカルロ法を用いてHimmelblauの最小値を探索する計算に制約式を適用します。

サンプルファイルの場所
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

サンプルファイルは ``sample/analytical/limitation`` にあります。
フォルダには以下のファイルが格納されています。

- ``input.toml``

  メインプログラムの入力ファイル。

- ``ref.txt``

  計算が正しく実行されたか確認するためのファイル (本チュートリアルを行うことで得られる ``best_result.txt`` の回答)。

- ``hist2d_limitation_sample.py``

  可視化のためのツール。
  
- ``do.sh``

  本チュートリアルを一括計算するために準備されたスクリプト

以下、これらのファイルについて説明したあと、実際の計算結果を紹介します。


入力ファイルの説明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

メインプログラム用の入力ファイル ``input.toml`` について説明します。
詳細については「入力ファイル」の章を参照してください。

.. code-block::

  [base]
  dimension = 2
  output_dir = "output"

  [algorithm]
  name = "exchange"
  seed = 12345

  [algorithm.param]
  max_list = [6.0, 6.0]
  min_list = [-6.0, -6.0]
  unit_list = [0.3, 0.3]

  [algorithm.exchange]
  Tmin = 1.0
  Tmax = 100000.0
  numsteps = 10000
  numsteps_exchange = 100

  [solver]
  name = "analytical"
  function_name = "himmelblau"

  [runner]
  [runner.limitation]
  co_a = [[1, -1],[1, 1]]
  co_b = [[0], [-1]]


``[base]`` セクションはメインプログラム全体のパラメータです。

- ``dimension`` は最適化したい変数の個数です。今の場合は2つの変数の最適化を行うので、 ``2`` を指定します。

- ``output_dir`` は出力先のディレクトリを指定します。  

``[algorithm]`` セクションでは、用いる探索アルゴリズムを設定します。

- ``name`` は使用するアルゴリズムの名前です。このチュートリアルでは交換モンテカルロ法を用いるので ``"exchange"`` を指定します。

- ``seed`` は擬似乱数生成器に与える乱数の種です。

``[algorithm.param]`` サブセクションは、最適化したいパラメータの範囲などを指定します。

- ``min_list`` と ``max_list`` はそれぞれ探索範囲の最小値と最大値を指定します。

- ``unit_list`` はモンテカルロ更新の際の変化幅(ガウス分布の偏差)です。

``[algorithm.exchange]`` サブセクションは、交換モンテカルロ法のハイパーパラメータを指定します。

- ``numstep`` はモンテカルロ更新の回数です。

- ``numsteps_exchange`` で指定した回数のモンテカルロ更新の後に、温度交換を試みます。

- ``Tmin``, ``Tmax`` はそれぞれ温度の下限・上限です。

- ``Tlogspace`` が ``true`` の場合、温度を対数空間で等分割します。指定がない場合のデフォルト値は ``true`` です。

``[solver]`` セクションではメインプログラムの内部で使用するソルバーとその設定を指定します。

- ``name`` は使用するソルバーの名前です。このチュートリアルでは ``analytical`` ソルバーに含まれる解析関数の解析を行います。

- ``function_name`` は ``analytical`` ソルバー内の関数名を指定します。

``[runner]`` セクションの ``[runner.limitation]`` サブセクションで制約式を設定します。
現在、制約式は :math:`N` 次元のパラメータ :math:`x` 、 :math:`M` 行 :math:`N` 列の行列 :math:`A` 、 
:math:`M` 次元の縦ベクトル :math:`b` から定義される :math:`Ax+b>0` の制約式が利用可能です。
パラメータとしては、以下の項目が設定可能です。

- ``co_a`` は行列 :math:`A` を設定します。

- ``co_b`` は縦ベクトル :math:`b` を設定します。

パラメータの詳しい設定方法はマニュアル内「入力ファイル」項の「 [``limitation``] セクション」を参照してください。
今回は

.. math::
  
  x_{1} - x_{2} > 0 \\
  x_{1} + x_{2} - 1 > 0

の制約式を課して実行しています。

計算の実行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

最初にサンプルファイルが置いてあるフォルダへ移動します。(以下、本ソフトウェアをダウンロードしたディレクトリ直下にいることを仮定します。)

.. code-block::

    $ cd sample/analytical/limitation

次に、メインプログラムを実行します。計算時間は通常のPCで20秒程度で終わります。

.. code-block::

    $ mpiexec -np 10 python3 ../../../src/odatse_main.py input.toml | tee log.txt

ここではプロセス数10のMPI並列を用いた計算を行っています。
Open MPI を用いる場合で、使えるコア数よりも要求プロセス数の方が多い時には、 ``mpiexec`` コマンドに ``--oversubscribe`` オプションを追加してください。

実行すると、 ``output`` フォルダが生成され、その中に各ランクのフォルダが作成されます。
更にその中には、各モンテカルロステップで評価したパラメータおよび目的関数の値を記した ``trial.txt`` ファイルと、実際に採択されたパラメータを記した ``result.txt`` ファイルが作成されます。
ともに書式は同じで、最初の2列がステップ数とプロセス内のwalker 番号、次が温度、3列目が目的関数の値、4列目以降がパラメータです。
以下は、 ``output/0/result.txt`` ファイルの冒頭部分です。

.. code-block::

    # step walker T fx x1 x2
    0 0 1.0 187.94429125133564 5.155393113805774 -2.203493345018569
    1 0 1.0 148.23606736778044 4.9995614992887525 -2.370212436322816
    2 0 1.0 148.23606736778044 4.9995614992887525 -2.370212436322816
    3 0 1.0 148.23606736778044 4.9995614992887525 -2.370212436322816
    ...

最後に、 ``output/best_result.txt`` に、目的関数が最小となったパラメータとそれを得たランク、モンテカルロステップの情報が書き込まれます。

.. code-block::

    nprocs = 10
    rank = 2
    step = 4523
    walker = 0
    fx = 0.00010188398524402734
    x1 = 3.584944906595298
    x2 = -1.8506985826548874

なお、一括計算するスクリプトとして ``do.sh`` を用意しています。
``do.sh`` では ``best_result.txt`` と ``ref.txt`` の差分も比較しています。
以下、説明は割愛しますが、その中身を掲載します。

.. code-block::

  #!/bin/bash

  mpiexec -np 10 --oversubscribe python3 ../../../src/odatse_main.py input.toml

  echo diff output/best_result.txt ref.txt
  res=0
  diff output/best_result.txt ref.txt || res=$?
  if [ $res -eq 0 ]; then
    echo TEST PASS
    true
  else
    echo TEST FAILED: best_result.txt and ref.txt differ
    false
  fi

計算結果の可視化
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``result.txt`` を図示して、制約式を満たした座標のみを探索しているかを確認します。
以下のコマンドを実行すると2次元パラメータ空間の図が ``<実行日>_histogram`` フォルダ内に作成されます。
生成されるヒストグラムは、burn-in期間として最初の1000ステップ分の探索を捨てたデータを使用しています。

.. code-block::

    $ python3 hist2d_limitation_sample.py -p 10 -i input.toml -b 0.1

作成された図には2本の直線 :math:`x_{1} - x_{2} = 0`, :math:`x_{1} + x_{2} - 1 = 0` と探索結果(事後確率分布のヒストグラム)を図示しています。
図を見ると :math:`x_{1} - x_{2} > 0`, :math:`x_{1} + x_{2} - 1 > 0` の範囲のみ探索をしていることが確認できます。
以下に図の一部を掲載します。

.. figure:: ../../../common/img/res_limitation.*

    サンプルされたパラメータと確率分布。横軸は ``x1`` , 縦軸は ``x2`` を表す。
