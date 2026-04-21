インストール
================================

実行環境・必要なパッケージ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- python 3.9 以上

  - 必要なpythonパッケージ

    - tomli (>= 1.2) : TOML形式の設定ファイルを読み込むため
    - numpy (>= 1.14) : 数値計算のため

  - Optional なパッケージ（特定の最適化手法を使用する場合に必要）

    - mpi4py (``mapper``, ``random_search``, ``exchange``, ``pamc`` などのMPI並列利用時) : 並列計算による高速化のため
    - scipy (Nelder-Mead法利用時) : Nelder-Mead法による最適化のため
    - physbo (ベイズ最適化利用時, ver. 2.0以上) : ベイズ最適化のため

ダウンロード・インストール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下記に示す方法で、 ``ODAT-SE`` python パッケージと ``odatse`` コマンドがインストールできます。

- PyPI からのインストール(推奨)

  - ``python3 -m pip install ODAT-SE``

    - ``--user`` オプションをつけるとローカル (``$HOME/.local``) にインストールできます

    - ``ODAT-SE[all]`` とすると Optional なパッケージも同時にインストールします

- ソースコードからのインストール

  1. ``git clone https://github.com/issp-center-dev/ODAT-SE``
  2. ``python3 -m pip install ./ODAT-SE``

  - ``pip`` のバージョンは 19 以上が必要です (``python3 -m pip install -U pip`` で更新可能)

- サンプルファイルのダウンロード

  - サンプルファイルはソースコードに同梱されています。
  - ``git clone https://github.com/issp-center-dev/ODAT-SE``

インストールの確認
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

インストールが正常に完了したかを確認するには、以下のコマンドを実行します：

.. code-block:: bash

  $ odatse --version
  $ odatse --help

アンインストール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ODAT-SE モジュールをアンインストールするには、以下のコマンドを実行します。

.. code-block:: bash

  $ python3 -m pip uninstall ODAT-SE

関連するオプションパッケージも個別にアンインストールする必要がある場合は、同様のコマンドで実行できます。

実行方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``odatse`` コマンドは定義済みの最適化アルゴリズム ``Algorithm`` と順問題ソルバー ``Solver`` の組み合わせで解析を行います。

.. code-block:: bash

  $ odatse input.toml

定義済みの ``Algorithm`` については :doc:`algorithm/index` を、
``Solver`` については :doc:`solver/index` を参照してください。

コマンドラインオプション
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``odatse`` コマンドには、実行モードを切り替えるためのオプションがあります。

- ``--init``

  初期状態から計算を開始します。デフォルトの動作です。

- ``--resume``

  チェックポイントファイルから中断時点の状態を復元して再開します。

- ``--cont``

  以前の計算結果を引き継いで、その続きから計算を進めます。

- ``--reset_rand``

  ``--resume`` または ``--cont`` と組み合わせて使用し、再開時に乱数系列を新しくします。

例:

.. code-block:: bash

  $ odatse --resume input.toml
  $ odatse --cont --reset_rand input.toml

2次元物質構造解析向けの順問題ソルバーを ODAT-SE から利用するためのラッパーが、独立なパッケージとして提供されています。
これらの解析を行う場合は、別途パッケージと順問題ソルバー本体をインストールしてください。
現在は以下のラッパーパッケージが用意されています。

- `odatse-STR <https://github.com/2DMAT/odatse-STR>`_ -- 全反射高速陽電子回折 (TRHEPD)
  表面構造解析のための高精度な手法です。

- `odatse-SXRD <https://github.com/2DMAT/odatse-SXRD>`_ -- 表面X線回折 (SXRD)
  表面や界面の原子配列を調べるためのX線回折手法です。

- `odatse-LEED <https://github.com/2DMAT/odatse-LEED>`_ -- 低速電子線回折 (LEED)
  固体表面の結晶構造を調べるための電子回折手法です。

``Algorithm`` や ``Solver`` をユーザーが準備する場合は、 ``ODAT-SE`` パッケージを利用します。
詳しくは :doc:`customize/index` を参照してください。

なお、 プログラムを改造する場合など、 ``odatse`` コマンドをインストールせずに直接実行することも可能です。
``src/odatse_main.py`` を利用してください。

.. code-block:: bash

  $ python3 src/odatse_main.py input.toml

MPI並列計算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ODAT-SEは、MPIを用いた並列計算をサポートしています。MPIを使用することで、複数のプロセスを用いて計算を高速化できます。

- ``mapper`` 、 ``random_search`` 、 ``exchange`` 、 ``pamc`` はMPI並列計算による高速化が可能です
- ``bayes`` は ``mpi4py`` が利用可能な環境では MPI 並列計算に対応します
- 並列実行時は、各プロセスがそれぞれ独自の乱数系列を持ちます (``seed`` と ``seed_delta`` パラメータ参照)
- チェックポイントファイルは各プロセスごとに作成されます

実行例:

.. code-block:: bash

  $ mpirun -np 4 odatse input.toml

``-np 4`` の部分は使用するプロセス数を指定します。使用可能なコア数に応じて調整してください。

環境によっては ``mpiexec`` や他のコマンド、またはジョブスケジューラを通してMPIプログラムを実行する場合もあります。特に大規模計算機センターなどでは、システム固有の実行方法があります。詳しくはご利用の環境のマニュアルを参照してください。

.. note::
  アルゴリズムによって並列化効率は異なります。例えば ``exchange`` では、レプリカ数と同じかそれ以下のプロセス数を使用するのが効率的です。
