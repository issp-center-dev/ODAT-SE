グリッド型探索
=====================================

ここでは、グリッド型探索を行い、回折データから原子座標を解析する方法について説明します。
グリッド型探索はMPIに対応しています。具体的な計算手順は ``minsearch`` の時と同様です。
ただし、探索グリッドを与えるデータ ``MeshData.txt`` を事前に準備する必要があります。

サンプルファイルの場所
~~~~~~~~~~~~~~~~~~~~~~~~

サンプルファイルは ``sample/sim-trhepd-rheed/single_beam/mapper`` にあります。
フォルダには以下のファイルが格納されています。

- ``bulk.txt``

  ``bulk.exe`` の入力ファイル

- ``experiment.txt`` , ``template.txt``

  メインプログラムでの計算を進めるための参照ファイル

- ``ref_ColorMap.txt``

  計算が正しく実行されたか確認するためのファイル(本チュートリアルを行うことで得られる ``ColorMap.txt`` の回答)。

- ``input.toml``

  メインプログラムの入力ファイル

- ``prepare.sh`` , ``do.sh``

  本チュートリアルを一括計算するために準備されたスクリプト

以下、これらのファイルについて説明したあと、実際の計算結果を紹介します。

参照ファイルの説明
~~~~~~~~~~~~~~~~~~~

``template.txt`` , ``experiment.txt`` については、
前のチュートリアル(Nealder-Mead法による最適化)と同じものを使用します。
ただし、計算を軽くするため ``value_03`` は用いずに ``3.5`` に固定し、
2次元のグリッド探索を行うように変更してあります。
実際に探索するグリッドは ``MeshData.txt`` で与えます。
サンプルでは ``MeshData.txt`` の中身は以下のようになっています。

.. code-block::

    1 6.000000 6.000000
    2 6.000000 5.750000
    3 6.000000 5.500000
    4 6.000000 5.250000
    5 6.000000 5.000000
    6 6.000000 4.750000
    7 6.000000 4.500000
    8 6.000000 4.250000
    9 6.000000 4.000000
    ...

1列目が通し番号、2列目以降は ``template.txt`` に入る``value_0`` , ``value_1`` の値が順に指定されています。

入力ファイルの説明
~~~~~~~~~~~~~~~~~~~

ここでは、メインプログラム用の入力ファイル ``input.toml`` について説明します。
``input.toml`` の詳細については入力ファイルに記載されています。
以下は、サンプルファイルにある ``input.toml`` の中身になります。

.. code-block::

    [base]
    dimension = 2

    [solver]
    name = "sim-trhepd-rheed"

    [solver.config]
    calculated_first_line = 5
    calculated_last_line = 74
    row_number = 2

    [solver.param]
    string_list = ["value_01", "value_02" ]
    degree_max = 7.0

    [solver.reference]
    path = "experiment.txt"
    first = 1
    last = 70

    [algorithm]
    name = "mapper"
    label_list = ["z1", "z2"]

最初に ``[base]`` セクションについて説明します。

- ``dimension`` は最適化したい変数の個数で、今の場合は ``template.txt`` で説明したように2つの変数の最適化を行うので、``2`` を指定します。

``[solver]`` セクションではメインプログラムの内部で使用するソルバーとその設定を指定します。

- ``name`` は使用したいソルバーの名前で、このチュートリアルでは、 ``sim-trhepd-rheed`` を用いた解析を行うので、 ``sim-trhepd-rheed`` を指定します。

ソルバーの設定は、サブセクションの ``[solver.config]``, ``[solver.param]``, ``[solver.reference]`` で行います。

``[solver.config]`` セクションではメインプログラム内部で呼び出す ``surf.exe`` により得られた出力ファイルを読み込む際のオプションを指定します。

- ``calculated_first_line`` は出力ファイルを読み込む最初の行数を指定します。

- ``calculated_last_line`` は出力ファイルを読み込む最後の行数を指定します。

- ``row_number`` は出力ファイルの何列目を読み込むかを指定します。

``[solver.param]`` セクションではメインプログラム内部で呼び出す ``surf.exe`` により得られた出力ファイルを読み込む際のオプションを指定します。

- ``string_list`` は、 ``template.txt`` で読み込む、動かしたい変数の名前のリストです。


- ``label_list`` は、``value_0x`` (x=1,2) を出力する際につけるラベル名のリストです。

- ``degree_max`` は、最大角度（度単位）の指定をします。

``[solver.reference]`` セクションでは、実験データの置いてある場所と読みこむ範囲を指定します。

- ``path`` は実験データが置いてあるパスを指定します。

- ``first`` は実験データファイルを読み込む最初の行数を指定します。

- ``end`` は実験データファイルを読み込む最後の行数を指定します。

``[algorithm]`` セクションでは、使用するアルゴリスムとその設定をします。

- ``name`` は使用したいアルゴリズムの名前で、このチュートリアルでは、グリッド探索による解析を行うので、 ``mapper`` を指定します。

- ``label_list`` は、``value_0x`` (x=1,2) を出力する際につけるラベル名のリストです。

その他、入力ファイルで指定可能なパラメータの詳細については入力ファイルの章をご覧ください。

計算実行
~~~~~~~~~~~~

最初にサンプルファイルが置いてあるフォルダへ移動します(以下、本ソフトウェアをダウンロードしたディレクトリ直下にいることを仮定します).

.. code-block::

    cd sample/sim-trhepd-rheed/single_beam/mapper

順問題の時と同様に、``bulk.exe`` と ``surf.exe`` をコピーします。

.. code-block::

    cp ../../../../../sim-trhepd-rheed/src/TRHEPD/bulk.exe .
    cp ../../../../../sim-trhepd-rheed/src/TRHEPD/surf.exe .

最初に ``bulk.exe`` を実行し、``bulkP.b`` を作成します。

.. code-block::

    ./bulk.exe

そのあとに、メインプログラムを実行します(計算時間は通常のPCで数秒程度で終わります)。

.. code-block::

    mpiexec -np 2 python3 ../../../../src/py2dmat_main.py input.toml | tee log.txt

ここではプロセス数2のMPI並列を用いた計算を行っています。
実行すると、各ランクのフォルダが作成され、その下に更にグリッドのidがついたサブフォルダ ``Log%%%%%``  ( ``%%%%%`` がグリッドのid)が作成されます
(``MeshData.txt`` に付けられた番号がグリッドのidとして割り振られます)。
以下の様な標準出力がされます。

.. code-block::

    Iteration : 1/33
    Read experiment.txt
    mesh before: [1.0, 6.0, 6.0]
    z1 =  6.00000
    z2 =  6.00000
    [' 6.00000', ' 6.00000']
    PASS : degree in lastline = 7.0
    PASS : len(calculated_list) 70 == len(convolution_I_calculated_list)70
    R-factor = 0.04785241875354398
    ...

``z1``, ``z2`` に各メッシュでの候補パラメータと、その時の``R-factor`` が出力されます。
最終的にグリッド上の全ての点で計算された ``R-factor`` は、``ColorMap.txt`` に出力されます。
今回の場合は

.. code-block::

    6.000000 6.000000 0.047852
    6.000000 5.750000 0.055011
    6.000000 5.500000 0.053190
    6.000000 5.250000 0.038905
    6.000000 5.000000 0.047674
    6.000000 4.750000 0.065919
    6.000000 4.500000 0.053675
    6.000000 4.250000 0.061261
    6.000000 4.000000 0.069351
    6.000000 3.750000 0.071868
    6.000000 3.500000 0.072739
    ...

のように得られます。1列目、2列目に ``value_01``, ``value_02`` の値が、3列目に ``R-factor`` が記載されます。
なお、一括計算するスクリプトとして ``do.sh`` を用意しています。
``do.sh`` では ``ColorMap.dat`` と ``ref_ColorMap.dat`` の差分も比較しています。
以下、説明は割愛しますが、その中身を掲載します。

.. code-block::

    sh prepare.sh

    ./bulk.exe

    time mpiexec -np 2 python3 ../../../../src/py2dmat_main.py input.toml

    echo diff ColorMap.txt ref_ColorMap.txt
    res=0
    diff ColorMap.txt ref_ColorMap.txt || res=$?
    if [ $res -eq 0 ]; then
      echo TEST PASS
      true
    else
      echo TEST FAILED: ColorMap.txt and ref_ColorMap.txt differ
      false
    fi

計算結果の可視化
~~~~~~~~~~~~~~~~~~~

``ColorMap.txt`` を図示することで、``R-factor`` の小さいパラメータがどこにあるかを推定することができます。
今回の場合は、以下のコマンドをうつことで2次元パラメータ空間の図 ``ColorMapFig.png`` が作成されます。

.. code-block::

    python3 plot_colormap_2d.py

作成された図を見ると、(5.25, 4.25) 付近に最小値を持っていることがわかります。

.. figure:: ../../../common/img/mapper.*

    2次元パラメータ空間上での ``R-factor`` 。

また、 ``RockingCurve.txt`` が各サブフォルダに格納されています。
これを用いることで、前チュートリアルの手順に従い、実験値との比較も行うことが可能です。
