``sxrd`` ソルバー
***********************************************

``sxrd`` は ``sxrdcalc`` を用いて原子位置 :math:`x` や原子の占有率、デバイワラー因子を与えることで Rocking curve を計算し、実験で得られた Rocking curve からの誤差を :math:`f(x)` として返す ``Solver`` です。

前準備
~~~~~~~~~~~~

``sxrdcalc`` は ``py2dmat`` から呼び出されます。
そのため、あらかじめ ``sxrdcalc`` をインストールしておく必要があります。
``sxrdcalc`` はGitHubの以下のURLで公開されています。

https://github.com/sxrdcalc/sxrdcalc

サイトにアクセスの上、「Code」-「Download zip」よりソースコード一式をダウンロードします。
zipファイル解凍後に、Makefileを自身の計算環境に合うように編集したあとに、 ``make`` コマンドを打つことで ``sxrdcalc`` の実行ファイルができます。

なお、 ``py2dmat`` を実行する前にあらかじめバルクデータを作成する必要があります(フォーマットについては、後述のソルバー用補助ファイルをご覧ください)。

入力パラメータ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``solver`` セクション中のサブセクション
``config``, ``post``, ``param``, ``reference`` を利用します。

[``config``] セクション
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``sxrd_exec_file``

  形式: string型

  説明: ソルバー ``sxrdcalc`` へのパス

- ``bulk_struc_in_file``

  形式: string型

  説明: バルク構造のインプットファイル。

以下、入力例を記載します。

.. code-block::

   [config]
   sxrd_exec_file = "../../sxrdcalc"
   bulk_struc_in_file = "sic111-r3xr3.blk"

[``param``] セクション
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``scale_factor``

  形式: float型 (default: 1.0)

  説明: ターゲットのRocking Curveとシミュレーションで得られるRocking Curveのスケールの値。

- ``opt_scale_factor``

  形式: bool型 (default: false)

  説明: ``scale_factor`` を最適化するかどうかのフラグ。
  
- ``type_vector``

  形式: list型

  説明: 最適化する変数の種類を正の数のlist形式で指定します。
  [``param.atom``]サブセクションで指定されるtypeの種類と、本リストが対応します。
  typeが同じ場合は、同じ変数として取り扱われます。


[``param.domain``] サブセクション
-----------------------------------
本セクションではドメインを作成します。作成したいドメイン分、定義する必要があります。
[``param.domain.atom``] サブサブセクションでドメイン内の情報を指定します。

- ``domain_occupancy``

  形式: float型

  説明: ドメイン全体の占有率。

[``param.domain.atom``] サブセクション
---------------------------------------------
本セクションはドメインに所属する最適化したい原子の個数分定義を行う必要があります。
なお、変数の種類を表すtypeは、正の数で入力します。

- ``name``

  形式: string型 (重複可)

  説明: 最適化する原子の名前。

- ``pos_center``

  形式: list型

  説明: 原子の中心座標。[:math:`x_0, y_0, z_0`]の形式で記載します(:math:`x_0, y_0, z_0` はfloat)。

- ``DWfactor``

  形式: float型

  説明: デバイワラー因子(単位は :math:`\text{Å}^{2}` )。

- ``occupancy``

  形式: float型 (default: 1.0)

  説明: 原子の占有率。

- ``displace_vector`` (省略可)

  形式: listのlist型

  説明: 原子を動かす方向を定義するベクトル。最大3方向まで指定可能。
       各リストで変位ベクトルと初期値を[type, :math:`D_{i1}, D_{i2}, D_{i3}` ]のようにして定義する(typeはint、:math:`D_{i1}, D_{i2}, D_{i3}` はfloat)。
       与えられた情報に従い、
       :math:`dr_i = (D_{i1} \vec{a} + D_{i2} \vec{b} + D_{i3} \vec{c}) * l_{type}` 
       のように :math:`l_{type}` を変位させます( :math:`\vec{a}, \vec{b}, \vec{c}` は ``bulk_struc_in_file`` もしくは ``struc_in_file`` で指定された入力ファイルに記載された単位格子ベクトルを表します)。
       
- ``opt_DW`` (省略可)

  形式: list型

  説明: デバイワラー因子を変化させる場合のスケールを設定します。
       [type, scale]のようにして定義されます。

- ``opt_occupancy``

  形式: int型

  説明: 定義した場合、占有率が変化する。指定された変数がtypeを表します。


以下、入力例を記載します。

.. code-block::

   [param]
   scale_factor = 1.0
   type_vector = [1, 2]

   [[param.domain]]
   domain_occupancy = 1.0
    [[param.domain.atom]]
      name = "Si"
      pos_center = [0.00000000, 0.00000000, 1.00000000]
      DWfactor = 0.0
      occupancy = 1.0
      displace_vector = [[1, 0.0, 0.0, 1.0]]
    [[param.domain.atom]]
      name = "Si"
      pos_center = [0.33333333, 0.66666667, 1.00000000]
      DWfactor = 0.0
      occupancy = 1.0
      displace_vector = [[1, 0.0, 0.0, 1.0]]
    [[param.domain.atom]]
      name = "Si"
      pos_center = [0.66666667, 0.33333333, 1.00000000]
      DWfactor = 0.0
      occupancy = 1.0
      displace_vector = [[1, 0.0, 0.0, 1.0]]
    [[param.domain.atom]]
      name = "Si"
      pos_center = [0.33333333, 0.33333333, 1.00000000]
      DWfactor = 0.0
      occupancy = 1.0
      displace_vector = [[2, 0.0, 0.0, 1.0]]
  

[``reference``] セクション
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``f_in_file``

  形式: string型

  説明: ターゲットとするロッキングカーブのインプットファイルへのパス。

  
ソルバー用補助ファイル
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ターゲット参照ファイル
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ターゲットにするデータが格納されたファイル。 [``reference``] セクションの ``f_in_file`` でパスを指定します。
1行ごとに ``h k l F sigma`` が出力されます。ここで、 ``h, k, l`` は波数, ``F`` は強度、 ``sigma`` は ``F`` の不確かさをそれぞれ表します。
以下、ファイル例を記載します。

.. code-block::
   
   0.000000 0.000000 0.050000 572.805262 0.1 
   0.000000 0.000000 0.150000 190.712559 0.1 
   0.000000 0.000000 0.250000 114.163340 0.1 
   0.000000 0.000000 0.350000 81.267319 0.1 
   0.000000 0.000000 0.450000 62.927325 0.1 
   ...

バルク構造ファイル
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

バルク構造のデータが格納されたファイル。 [``config``] セクションの ``bulk_struc_in_file`` でパスを指定します。
1行目がコメント, 2行目が ``a b c alpha beta gamma`` を表します。
ここで、 ``a`` , ``b``, ``c`` はユニットセルの格子定数、 ``alpha``, ``beta``, ``gamma`` はそれらのなす角です。
3行目以降は ``atomsymbol r1 r2 r3 DWfactor occupancy`` を指定します。
ここで、 ``atomsymbol`` は原子種、 ``r1``, ``r2``, ``r3`` は原子の位置座標、 ``DWfactor`` はデバイワラー因子、 ``occupancy`` は占有率をそれぞれ表します。
以下、ファイル例を記載します。

.. code-block::

   # SiC(111) bulk
   5.33940 5.33940  7.5510487  90.000000 90.000000 120.000000
   Si 0.00000000   0.00000000   0.00000000 0.0 1.0
   Si 0.33333333   0.66666667   0.00000000 0.0 1.0
   Si 0.66666667   0.33333333   0.00000000 0.0 1.0
   C  0.00000000   0.00000000   0.25000000 0.0 1.0
   ...
   
出力ファイル
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sxrd`` では、 計算時に出力されるファイルが、ランクの番号が記載されたフォルダ下に一式出力されます。
ここでは、 ``py2dmat`` で独自に出力するファイルについて説明します。

``stdout``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``sxrd`` が出力する標準出力が記載されています。
sxrdのLeast square fittingに対して、初期パラメータとして変数を与え、1ショット計算(iteration数=0)をした際のRfactorを計算します。
RfactorはFit results以下のRに記載されます。
以下、出力例です。

.. code-block::

    ---------------------------------------
    Program py2dmat/mapper_sxrd/sxrdcalc for surface x-ray diffraction calculations.
    Version 3.3.3 - August 2019


     Inputfile: lsfit.in
    Least-squares fit of model to experimental structure factors.

    ...

    Fit results:
    Fit not converged after 0 iterations.
    Consider increasing the maximum number of iterations or find better starting values.
    chi^2 = 10493110.323318, chi^2 / (degree of freedom) = 223257.666454 (Intensities)
    chi^2 = 3707027.897897, chi^2 / (degree of freedom) = 78872.933998 (Structure factors)
    R = 0.378801

    Scale factor:   1.00000000000000 +/- 0.000196
    Parameter Nr. 1:   3.500000 +/- 299467640982.406067
    Parameter Nr. 2:   3.500000 +/- 898402922947.218384

    Covariance matrix:
              0            1            2
     0  0.0000000383 20107160.3315223120 -60321480.9945669472
     1  20107160.3315223120 89680867995567253356544.0000000000 -269042603986701827178496.0000000000
     2  -60321480.9945669472 -269042603986701827178496.0000000000 807127811960105615753216.0000000000

