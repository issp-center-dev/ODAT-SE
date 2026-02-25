plt_model_evidence.py
=====================

NAME
----
model evidence を計算する

SYNOPSIS
--------

.. code-block:: bash

   python3 plt_model_evidence.py [OPTION]... -n NDATA FILEs


DESCRIPTION
-----------

PAMC の出力ファイル FILE から beta および分配関数の値を取り出し、model evidence を計算する。
結果は標準出力に書き出される。また、結果をプロットした図をファイルに出力する。

複数の FILE を指定した場合は、それらの model evidence の平均と分散を出力し、エラーバー付きのプロットを生成する。

.. note::
   * Python 3.6以上が必要です(f文字列を使用しているため)。
   * 計算はすべて対数スケールで行われるため、数値的に安定しています。
   * プロットのx軸(beta)は常に対数スケールで表示されます。

**FILE**
    PAMCの出力ファイル名 (fx.txt)。複数のファイルを指定可能。

**-n NDATA, --ndata NDATA**
    各データセットに含まれるデータ点の数をカンマ区切りの整数値で指定する。必須パラメータ。例：「100」(1つのデータセットに100点)、「50,100,75」(3つのデータセットにそれぞれ50点、100点、75点)

**-w WEIGHT, --weight WEIGHT**
    データセット間の相対重みをカンマ区切りの数値で指定する。重みは和が 1.0 になるように自動で規格化される。重みの数値の個数とデータ点の個数は一致させる必要がある。

**-V VOLUME, --Volume VOLUME**
    事前確率分布の normalization (定義域の体積 :math:`V_\Omega`) を指定する。デフォルトは 1.0。

**-f RESULT, --result RESULT**
    model evidence の値を出力するファイル名。デフォルトは model_evidence.txt 。

**-o OUTPUT, --output OUTPUT**
    model evidence のプロットを出力するファイル名。出力形式は拡張子を元に設定され、matplotlib がサポートする形式を指定可能。デフォルトは model_evidence.png 。

**-h, --help**
    ヘルプメッセージを表示してプログラムを終了する。

USAGE
-----

1. 基本的な使用方法(1つのデータファイルと1つのデータセット)

   .. code-block:: bash

      $ python3 plt_model_evidence.py -n 100 fx.txt

   データ点数100のデータセットについて model evidence を計算し、
   model_evidence.txt と model_evidence.png を出力する。

2. 複数のデータセットがある場合

   .. code-block:: bash

      $ python3 plt_model_evidence.py -n 50,100,75 -w 0.2,0.5,0.3 fx.txt

   3つのスポット(データ点数がそれぞれ50、100、75で、相対重みが0.2、0.5、0.3)について、
   model evidence を計算する。

3. 複数のデータファイルを使用する場合

   .. code-block:: bash

      $ python3 plt_model_evidence.py -n 100 -o evidence_plot.pdf -f evidence_data.txt fx_1.txt fx_2.txt fx_3.txt

   3つのデータファイルから model evidence を計算し、それらの平均と分散を求める。
   結果を evidence_data.txt に出力し、evidence_plot.pdf にエラーバー付きのプロットを生成する。

NOTES
-----

model evidence の計算
~~~~~~~~~~~~~~~~~~~~~
R-factor を以下のように定義する:

.. math::

   R(X;D)^2 = \sum_\mu w_\mu \sum_i \left( I_\mu(\theta_i) - I^{\text{(cal)}}_\mu(\theta_i;X) \right)^2

:math:`I_\mu(\theta_i)` はデータセット :math:`\mu` に含まれる計測データ点、 :math:`I^{\text{(cal)}}_\mu(\theta_i;X)` はパラメータ :math:`X` のもとでの理論計算値とする。
:math:`w_\mu` はデータセットの相対重みで、合計が1になるよう正規化されている。

model evidence :math:`P(D|\beta)` は以下の式で計算される:

.. math::

   \log P(D|\beta) = \log Z(D;\beta) - \log V_\Omega + \frac{n}{2} \log \beta + \sum_{\mu} \frac{n_{\mu}}{2} \log w_{\mu} - \frac{n}{2} \log \pi

ここで、 :math:`Z(D;\beta)` は分配関数

.. math::

   Z(D;\beta) = \int_\Omega \exp\left(-\beta\,R(X;D)^2\right) dX


また、

 * :math:`V_\Omega`: 事前確率分布の正規化因子
 * :math:`n_\mu`: 各データセットのデータ点数
 * :math:`n`: 全データ点数(すべてのデータセットの合計)
 * :math:`\beta`: 逆温度

とする。

入力ファイルの形式
~~~~~~~~~~~~~~~~~~

入力ファイル(PAMCの出力ファイル)は以下の形式を想定しています:

.. code-block:: text

   # コメント行(任意)
   beta_value  fx_mean  fx_var  nreplica  logz_value  acceptance
   ...

スクリプトは各行から以下の値を読み取ります:
 * 第1列(インデックス0): beta値(逆温度)
 * 第5列(インデックス4): logz値(対数分配関数)

出力ファイルの形式
~~~~~~~~~~~~~~~~~~

出力ファイル(model_evidence.txt)の形式は以下の通りです:

.. code-block:: text

   # max log_P(D;beta) = {最大値} at Tstep = {インデックス}, beta = {対応するbeta値}
   # $1: Tstep
   # $2: beta
   # $3: model_evidence
   0  beta0  model_evidence0
   1  beta1  model_evidence1
   ...

複数の入力ファイルを処理した場合は、分散の列が追加されます:

.. code-block:: text

   # max log_P(D;beta) = {最大値} at Tstep = {インデックス}, beta = {対応するbeta値}
   # $1: Tstep
   # $2: beta
   # $3: average model_evidence
   # $4: variance
   0  beta0  avg_model_evidence0  variance0
   1  beta1  avg_model_evidence1  variance1
   ...

処理の仕組み
~~~~~~~~~~~~

このスクリプトは以下の手順で処理を行います:

1. 入力ファイルからbeta値とlogz値を読み込む
2. 各データセットのデータ点数と重みを取得
3. model evidenceの対数値を計算
4. 複数ファイルの場合は平均と分散を計算
5. 結果をファイルに出力
6. model evidenceをbetaの関数としてプロット

プロットの特性
~~~~~~~~~~~~~~

* X軸(beta)は常に対数スケールで表示
* 単一ファイルの場合は点のみ、複数ファイルの場合はエラーバー付きで表示
* マーカーは赤色の「x」で表示
* グリッド線を表示し、データの位置を把握しやすくする

エラー処理
~~~~~~~~~~

* 入力ファイルが存在しない場合: ファイルオープンエラーが発生
* データ形式が不正: numpy.loadtxtでエラーが発生
* NDATAとWEIGHTの長さが一致しない場合: AssertionErrorが発生

特に、データ点数のリストとそれらの重みの数は必ず一致させる必要があります。

