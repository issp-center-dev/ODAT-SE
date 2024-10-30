ポピュレーションアニーリングモンテカルロ法 ``pamc``
=========================================================

``pamc`` はポピュレーションアニーリングモンテカルロ法を用いてパラメータ探索を行う ``Algorithm`` です。

前準備
~~~~~~~~

MPI 並列をする場合にはあらかじめ `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ をインストールしておく必要があります。::

  python3 -m pip install mpi4py

入力パラメータ
~~~~~~~~~~~~~~~~~~~

サブセクション ``param`` と ``pamc`` を持ちます。

[``param``] セクション
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

探索空間を定義します。
``mesh_path`` キーが存在する場合は離散空間を、そうでない場合は連続空間を探索します。

- 連続空間

  - ``initial_list``

    形式: 実数型のリスト。長さはdimensionの値と一致させます。

    説明: パラメータの初期値。 定義しなかった場合は一様ランダムに初期化されます。

  - ``unit_list``

    形式: 実数型のリスト。長さはdimensionの値と一致させます。

    説明: 各パラメータの単位。
          探索アルゴリズム中では、各パラメータをそれぞれこれらの値で割ることで、
          簡易的な無次元化・正規化を行います。
          定義しなかった場合にはすべての次元で 1.0 となります。

  - ``min_list``

    形式: 実数型のリスト。長さはdimensionの値と一致させます。

    説明: パラメータが取りうる最小値。
          モンテカルロ探索中にこの値を下回るパラメータが出現した場合、
          ソルバーは評価されずに、値が無限大だとみなされます。

  - ``max_list``

    形式: 実数型のリスト。長さはdimensionの値と一致させます。

    説明: パラメータが取りうる最大値。  
          モンテカルロ探索中にこの値を上回るパラメータが出現した場合、
          ソルバーは評価されずに、値が無限大だとみなされます。

- 離散空間

  - ``mesh_path``

    形式: ファイルパス

    説明: メッシュ定義ファイル。

  - ``neighborlist_path``

    形式: ファイルパス

    説明: 近傍リスト定義ファイル。


[``pamc``] セクション
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``numsteps``

  形式: 整数値。

  説明: モンテカルロ更新を行う総回数。

- ``numsteps_annealing``

  形式: 整数値。

  説明: 「温度」を下げる頻度。この回数モンテカルロ更新を行った後に温度が下がります。

- ``numT``

  形式: 整数値。

  説明: 「温度」点の数。

- ``Tmin``

  形式: 実数値。

  説明: 「温度」(:math:`T`)の最小値。

- ``Tmax``

  形式: 実数値。

  説明: 「温度」(:math:`T`)の最大値。

- ``bmin``

  形式: 実数値。

  説明: 「逆温度」(:math:`\beta = 1/T`)の最小値。
  温度と逆温度はどちらか片方だけを指定する必要があります。

- ``bmax``

  形式: 実数値。

  説明: 「逆温度」(:math:`\beta = 1/T`)の最大値。
  温度と逆温度はどちらか片方だけを指定する必要があります。

- ``Tlogspace``

  形式: 真偽値。 (default: true)

  説明: 「温度」を各レプリカに割り当てる際に、対数空間で等分割するか否かを指定します。
        true のときは対数空間で等分割します。

- ``nreplica_per_proc``

  形式: 整数。 (default: 1)

  説明: ひとつのMPI プロセスが担当するレプリカの数。

- ``resampling_interval``

  形式: 整数。 (default: 1)

  説明: レプリカのリサンプリングを行う頻度。この回数だけ温度降下を行った後にリサンプリングが行われます。

- ``fix_num_replicas``

  形式: 真偽値。 (default: true)

  説明: リサンプリングの際、レプリカ数を固定するかどうか。

ステップ数について
********************

``numsteps``, ``numsteps_annealing``, ``numT`` の3つのうち、どれか2つを同時に指定してください。
残りの1つは自動的に決定されます。

アルゴリズム補助ファイル
~~~~~~~~~~~~~~~~~~~~~~~~~~

メッシュ定義ファイル
^^^^^^^^^^^^^^^^^^^^^^^^^^

本ファイルで探索するグリッド空間を定義します。
1列目にメッシュのインデックス（実際には使用されません）、
2列目以降は探索空間の座標を指定します。

以下、サンプルを記載します。

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


近傍リスト定義ファイル
^^^^^^^^^^^^^^^^^^^^^^^^^^

離散空間をモンテカルロ法で探索する場合、各点 :math:`i` ごとに次に移動できる点 :math:`j` を定めておく必要があります。
そのために必要なのが近傍リスト定義ファイルです。

1列目に始点の番号 :math:`i` を記載し、
2列目以降に :math:`i` から移動できる終点 :math:`j` を列挙します。

近傍リスト定義ファイルをメッシュ定義ファイルから生成するツール ``py2dmat_neighborlist`` が提供されています。
詳細は :doc:`../tool` を参照してください。

.. code-block::

    0 1 2 3
    1 0 2 3 4
    2 0 1 3 4 5
    3 0 1 2 4 5 6 7
    4 1 2 3 5 6 7 8
    5 2 3 4 7 8 9
    ...

出力ファイル
~~~~~~~~~~~~~~~~~~~~~

``RANK/trial_T#.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

各温度点(``#``) ごと、モンテカルロサンプリングで提案されたパラメータと、対応する目的関数の値です。
1列目にステップ数、2列目にプロセス内のwalker 番号、3列目にレプリカの逆温度、4列目に目的関数の値、5列目からパラメータが記載されます。
最後の2列はそれぞれレプリカの重み (Neal-Jarzynski weight) と祖先（計算開始時のレプリカ番号）です。

.. code-block::

    # step walker beta fx x1 weight ancestor
    0 0 0.0 73.82799488298886 8.592321856342956 1.0 0
    0 1 0.0 13.487174782058675 -3.672488908364282 1.0 1
    0 2 0.0 39.96292704464803 -6.321623766458111 1.0 2
    0 3 0.0 34.913851603463 -5.908794428939206 1.0 3
    0 4 0.0 1.834671825646121 1.354500581633733 1.0 4
    0 5 0.0 3.65151610695736 1.910894059585031 1.0 5
    ...


``RANK/trial.txt``
^^^^^^^^^^^^^^^^^^^^^

``trial_T#.txt`` をすべてまとめたものです。

``RANK/result_T#.txt``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
各温度点、モンテカルロサンプリングで生成されたパラメータと、対応する目的関数の値です。
``trial.txt`` と同一の書式です。

.. code-block::

    # step walker beta fx x1 weight ancestor
    0 0 0.0 73.82799488298886 8.592321856342956 1.0 0
    0 1 0.0 13.487174782058675 -3.672488908364282 1.0 1
    0 2 0.0 39.96292704464803 -6.321623766458111 1.0 2
    0 3 0.0 34.913851603463 -5.908794428939206 1.0 3
    0 4 0.0 1.834671825646121 1.354500581633733 1.0 4
    0 5 0.0 3.65151610695736 1.910894059585031 1.0 5
    ...

``RANK/result.txt``
^^^^^^^^^^^^^^^^^^^^^

``result_T#.txt`` をすべてまとめたものです。


``best_result.txt``
^^^^^^^^^^^^^^^^^^^^

サンプリングされた全データのうち、目的関数の値が最小となったパラメータと、対応する目的関数の値です。

.. code-block::

    nprocs = 4
    rank = 2
    step = 65
    fx = 0.008233957976993406
    z1 = 4.221129370933539
    z2 = 5.139591716517661


``fx.txt``
^^^^^^^^^^^^^^

各温度ごとに、全レプリカの情報をまとめたものです。
1列目は逆温度が、2列目と3列目には目的関数の期待値およびその標準誤差が、4列目にはレプリカの総数が、5列目には規格化因子（分配関数）の比の対数

.. math::

   \log\frac{Z}{Z_0} = \log\int \mathrm{d}x e^{-\beta f(x)} - \log\int \mathrm{d}x e^{-\beta_0 f(x)}

が、6列目にはモンテカルロ更新の採択率が出力されます。
ここで :math:`\beta_0` は計算している :math:`\beta` の最小値です。

.. code-block::

    # $1: 1/T
    # $2: mean of f(x)
    # $3: standard error of f(x)
    # $4: number of replicas
    # $5: log(Z/Z0)
    # $6: acceptance ratio
    0.0 33.36426034198166 3.0193077565358273 100 0.0 0.9804
    0.1 4.518006242920819 0.9535301415484388 100 -1.2134775491597027 0.9058
    0.2 1.5919146358616842 0.2770369776964151 100 -1.538611313376179 0.9004
    ...

アルゴリズム解説
~~~~~~~~~~~~~~~~~~~

問題と目的
^^^^^^^^^^^^

分布パラメータ :math:`\beta_i` のもとでの配位 :math:`x` の重みを
:math:`f_i(x)` と書くと（例えばボルツマン因子 :math:`f_i(x) = \exp\left[-\beta_i E(x)\right]`\ ）、
:math:`A` の期待値は

.. math::

   \langle A\rangle_i
   = \frac{\int \mathrm{d}xA(x)f_i(x)}{\int \mathrm{d}x f_i(x)}
   = \frac{1}{Z}\int \mathrm{d}xA(x)f_i(x)
   = \int \mathrm{d}xA(x)\tilde{f}_i(x)

とかけます。
ここで :math:`Z = \int \mathrm{d} x f_i(x)` は規格化因子（分配関数）で、 :math:`\tilde{f}(x) = f(x)/Z` は配位 :math:`x` の確率密度です。

目的は複数の分布パラメータについてこの期待値および規格化因子（の比）を数値的に求めることです。

Annealed Importance Sampling (AIS) [1]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

次の同時確率分布

.. math::

   \tilde{f}(x_0, x_1, \dots, x_n) = \tilde{f}_n(x_n) \tilde{T}_n(x_n, x_{n-1}) \tilde{T}_{n-1}(x_{n-1}, x_{n-2}) \cdots \tilde{T}_1(x_1, x_0)

を満たす点列 :math:`\{x_i\}` を考えます。ここで

.. math::

   \tilde{T}_i(x_i, x_{i-1}) = T_i(x_{i-1}, x_i) \frac{\tilde{f}_i(x_{i-1})}{\tilde{f}_i(x_i)}

であり、 :math:`T_i(x, x')` は :math:`\beta_i` のもとでの配位 :math:`x`
から :math:`x'` への遷移確率で、釣り合い条件

.. math::


   \int \mathrm{d}x \tilde{f}_i(x) T_i(x, x') = \tilde{f}_i(x')

を満たすようにとります（つまりは普通のMCMCにおける遷移確率行列）。

.. math::


   \int \mathrm{d} x_{i-1} \tilde{T}_i(x_i, x_{i-1})
   = \int \mathrm{d} x_{i-1} \tilde{f}_i(x_{i-1}) T_i(x_{i-1}, x_i) / \tilde{f}_i(x_i)
   = 1

となるので、 :math:`\tilde{f}_n(x_n)` は
:math:`\tilde{f}(x_0, x_1, \dots, x_n)` の周辺分布

.. math::


   \tilde{f}_n(x_n) = \int \prod_{i=0}^{n-1} \mathrm{d} x_i \tilde{f}(x_0, x_1, \dots, x_n)

です。
これを利用すると、 :math:`\tilde{f}_n` における平均値 :math:`\langle A \rangle_n` は拡張した配位の重み付き平均として

.. math::


   \begin{split}
   \langle A \rangle_n
   &\equiv
   \int \mathrm{d} x_n A(x_n) \tilde{f}_n(x_n) \\
   &= \int \prod_i \mathrm{d} x_i A(x_n) \tilde{f}(x_0, x_1, \dots, x_n)
   \end{split}

と表せます。

さて、残念ながら :math:`\tilde{f}(x_0, x_1, \dots, x_n)`
に従うような点列を直接生成することは困難です。そこでもっと簡単に、

1. 確率 :math:`\tilde{f}_0(x)` に従う :math:`x_0` を生成する

   -  例えば MCMC を利用する

2. :math:`x_i` から :math:`T_{i+1}(x_i, x_{i+1})` によって :math:`x_{i+1}` を生成する

   - :math:`T_{i+1}` は釣り合い条件を満たすような遷移確率行列なので、普通にMCMCを行えば良い

という流れに従って点列 :math:`\{x_i\}` を生成すると、これは同時確率分布

.. math::

   \tilde{g}(x_0, x_1, \dots, x_n) = \tilde{f}_0(x_0) T_1(x_0, x_1) T_2(x_1, x_2) \dots T_n(x_{n-1}, x_n)

に従います。これを利用すると期待値 :math:`\langle A \rangle_n` は

.. math::


   \begin{split}
   \langle A \rangle_n
   &= \int \prod_i \mathrm{d} x_i A(x_n) \tilde{f}(x_0, x_1, \dots, x_n) \\
   &= \int \prod_i \mathrm{d} x_i A(x_n) \frac{\tilde{f}(x_0, x_1, \dots, x_n)}{\tilde{g}(x_0, x_1, \dots, x_n)} \tilde{g}(x_0, x_1, \dots, x_n) \\
   &= \left\langle A\tilde{f}\big/\tilde{g} \right\rangle_{g, n}
   \end{split}

と評価できます (reweighting method)。
:math:`\tilde{f}` と :math:`\tilde{g}` との比は、

.. math::


   \begin{split}
   \frac{\tilde{f}(x_0, \dots, x_n)}{\tilde{g}(x_0, \dots, x_n)}
   &= 
   \frac{\tilde{f}_n(x_n)}{\tilde{f}_0(x_0)}
   \prod_{i=1}^n \frac{\tilde{T}_i(x_i, x_{i-1})}{T(x_{i-1}, x_i)} \\
   &=
   \frac{\tilde{f}_n(x_n)}{\tilde{f}_0(x_0)}
   \prod_{i=1}^n \frac{\tilde{f}_i(x_{i-1})}{\tilde{f}_i(x_i)} \\
   &=
   \frac{Z_0}{Z_n}
   \frac{f_n(x_n)}{f_0(x_0)}
   \prod_{i=1}^n \frac{f_i(x_{i-1})}{f_i(x_i)} \\
   &=
   \frac{Z_0}{Z_n}
   \prod_{i=0}^{n-1} \frac{f_{i+1}(x_{i})}{f_i(x_i)} \\
   &\equiv
   \frac{Z_0}{Z_n} w_n(x_0, x_1, \dots, x_n)
   \end{split}

とかけるので、期待値は

.. math::

   \langle A \rangle_n = \left\langle A\tilde{f}\big/\tilde{g} \right\rangle_{g, n}
   = \frac{Z_0}{Z_n} \langle Aw_n \rangle_{g,n}

となります。
規格化因子の比 :math:`Z_n/Z_0` は :math:`\langle 1 \rangle_n = 1` を用いると

.. math::

   \frac{Z_n}{Z_0} = \langle w_n \rangle_{g,n}

と評価できるので、 :math:`A` の期待値は

.. math::

   \langle A \rangle_n = \frac{\langle Aw_n \rangle_{g,n}}{\langle w_n \rangle_{g,n}}

という、重み付き平均の形で評価できます。
この重み :math:`w_n` を Neal-Jarzynski 重みと呼びます。

population annealing (PA) [2]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AIS を使うと各 :math:`\beta` に対する期待値を重み付き平均という形で計算できますが、
:math:`\beta` の幅が大きくなると重み :math:`w` の分散が大きくなってしまいます。
そのため、適当な周期で確率 :math:`p^{(k)} = w^{(k)} / \sum_k w^{(k)}` に従いレプリカをリサンプリングし、
レプリカに割当られた重みをリセット :math:`(w=1)` します。

PAMC のアルゴリズムは次の擬似コードで示されます:

.. code-block:: python

    for k in range(K):
        w[0, k] = 1.0
        x[0, k] = draw_from(β[0])
    for i in range(1, N):
        for k in range(K):
            w[i, k] = w[i-1, k] * ( f(x[i-1,k], β[i]) / f(x[i-1,k], β[i-1]) )
        if i % interval == 0:
            x[i, :] = resample(x[i, :], w[i, :])
            w[i, :] = 1.0
        for k in range(K):
            x[i, k] = transfer(x[i-1, k], β[i])
        a[i] = sum(A(x[i,:]) * w[i,:]) / sum(w[i,:])

リサンプリング手法として、レプリカ数を固定する方法[2]と固定しない方法[3]の2通りがあります。

参考文献
^^^^^^^^^^^^^

[1] R. M. Neal, Statistics and Computing **11**, 125-139 (2001).

[2] K. Hukushima and Y. Iba, AIP Conf. Proc. **690**, 200 (2003).

[3] J. Machta, PRE **82**, 026704 (2010).
