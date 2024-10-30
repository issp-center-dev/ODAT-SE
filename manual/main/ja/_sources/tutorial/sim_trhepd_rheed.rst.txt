TRHEPD 順問題ソルバー
========================

2DMAT は順問題ソルバーのひとつとして、 反射高速(陽)電子回折(RHEED, TRHEPD)の強度計算 (A. Ichimiya, Jpn. J. Appl. Phys. 22, 176 (1983); 24, 1365 (1985)) 
を行うプログラム `sim-trhepd-rheed <https://github.com/sim-trhepd-rheed/sim-trhepd-rheed/>`_ のラッパーを提供しています。
本チュートリアルでは sim-trhepd-rheed を用い、様々なアルゴリズムを利用した解析を行います。
最初に、チュートリアルを行うために必要な sim-trhepd-rheed のインストールおよびテストを行います。

ダウンロード・インストール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

まずチュートリアルの前提として、 ``2DMAT`` フォルダがある場所にいることを仮定します。::

   $ ls -d 2DMAT
   2DMAT/

GitHub の sim-trhepd-rheed リポジトリから、ソースコード一式を入手し、ビルドします。::

   git clone http://github.com/sim-trhepd-rheed/sim-trhepd-rheed
   cd sim-trhepd-rheed/src
   make

makeが成功すると、 ``bulk.exe`` 及び  ``surf.exe`` が作成されます。
		

計算実行
~~~~~~~~~~

sim-trhepd-rheed では、最初に ``bulk.exe`` で表面構造のバルク部分に関する計算をします。
その後、 ``bulk.exe`` の計算結果 (``bulkP.b`` ファイル) を用いて、 ``surf.exe`` 表面構造の表面部分を計算します。

このチュートリアルでは実際に、 TRHEPD 計算をしてみます。
サンプルとなる入力ファイルは2DMAT の ``sample/sim-trhepd-rheed`` にあります。
まず、このフォルダを適当な作業用フォルダ ``work`` にコピーします。

.. code-block::

   cd ../../
   cp -r 2DMAT/sample/sim-trhepd-rheed/solver work
   cd work

次に ``bulk.exe`` と ``surf.exe`` を ``work`` にコピーします

.. code-block::

   cp ../sim-trhepd-rheed/src/bulk.exe .
   cp ../sim-trhepd-rheed/src/surf.exe .

``bulk.exe`` を実行します。

.. code-block::

   ./bulk.exe

上記実行時に、以下のように出力され、バルクファイル ``bulkP.b`` が生成されます。

.. code-block::

   0:electron 1:positron ?
   P
   input-filename (end=e) ? :
   bulk.txt
   output-filename :
   bulkP.b

続いて、 ``surf.exe`` を実行します。

.. code-block::

   ./surf.exe

上記実行時に、以下のように出力されます。

.. code-block::

   bulk-filename (end=e) ? :
   bulkP.b
   structure-filename (end=e) ? :
   surf.txt
   output-filename :
   surf-bulkP.md
   surf-bulkP.s

実行後に、ファイル ``surf-bulkP.md`` 、 ``surf-bulkP.s``
及び ``SURFYYYYMMDD-HHMMSSlog.txt`` が生成されます。
(YYYYMMDD、 HHMMSS には実行日時に対応した数字が入ります)

計算結果の可視化
~~~~~~~~~~~~~~~~~

``surf-bulkP.s`` は以下の通りです。

.. code-block::

   #azimuths,g-angles,beams
   1 56 13
   #ih,ik
   6 0 5 0 4 0 3 0 2 0 1 0 0 0 -1 0 -2 0 -3 0 -4 0 -5 0 -6 0
   0.5000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.1595E-01, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
   0.6000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.1870E-01, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
   0.7000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.2121E-01, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
   0.8000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.2171E-02, 0.1927E-01, 0.2171E-02, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
   0.9000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.4397E-02, 0.1700E-01, 0.4397E-02, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
   0.1000E+01, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.6326E-02, 0.1495E-01, 0.6326E-02, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00, 0.0000E+00,
   (以下略)

上記ファイルより、縦軸に角度（5行目以降の1列目データ）と(0,0)ピークの強度（5行目以降の8列目データ）から
ロッキングカーブを作成します。
Gnuplot等のグラフソフトを用いる事も出来ますが、ここでは、 ``2DMAT/script`` フォルダにあるプログラム ``plot_bulkP.py`` を用います。
以下のように実行して下さい。

.. code-block::

   python3 ../2DMAT/script/plot_bulkP.py

以下のような ``plot_bulkP.png`` が作成されます。

.. figure:: ../../../common/img/plot_bulkP.*

   Si(001)-2x1面のロッキングカーブ。

この00ピークの回折強度のデータに対し、コンボリューションを掛けたうえで規格化します。
``surf-bulkP.s`` を準備して、 ``make_convolution.py`` を実行してください。

.. code-block::

   python3 ../2DMAT/script/make_convolution.py

実行すると、以下のようなファイル ``convolution.txt`` ができあがります。

.. code-block::

   0.500000 0.010818010
   0.600000 0.013986716
   0.700000 0.016119093
   0.800000 0.017039022
   0.900000 0.017084666
   （中略）
   5.600000 0.000728539
   5.700000 0.000530758
   5.800000 0.000412908
   5.900000 0.000341740
   6.000000 0.000277553

1列目が視射角、2列目が ``surf-bulkP.s`` に書かれた00ピーク回折強度のデータに
半値幅0.5のコンボリューションを付加して規格化したものです。
