odatse コマンド
================

NAME
----
odatse - 探索アルゴリズムと順問題ソルバーを組み合わせて逆問題解析を実行する

SYNOPSIS
--------

.. code-block:: bash

   odatse [-h] [--version] [--init | --resume | --cont] [--reset_rand]
          [--nalg NALG] [--nsolve NSOLVE] inputfile

DESCRIPTION
-----------

TOML 形式の入力ファイルを読み込み、 ``[algorithm]`` セクションで指定された探索アルゴリズムと ``[solver]`` セクションで指定された順問題ソルバーを組み合わせて逆問題解析を実行する。
入力ファイルの仕様は :doc:`/input/index` を、出力ファイルは :doc:`/output` を参照。

MPI 並列で実行する場合は ``mpiexec`` から起動する。

.. code-block:: bash

   mpiexec -np N odatse [OPTION]... inputfile

指定可能なコマンドラインオプションを以下に示す。

**inputfile**
    TOML 形式の入力ファイル。

**--init**
    初期状態から計算を開始する。デフォルトの動作。

**--resume**
    チェックポイントファイルから中断時点の状態を復元して再開する。
    ``[algorithm]`` セクションでチェックポイント機能 (``checkpoint = true``) を有効にして実行した計算が対象となる。

**--cont**
    終了した計算の結果を引き継ぎ、続きから計算を進める。
    ステップ数や温度点数を増やして計算を延長する場合に使用する。
    ``--resume`` と同様に、直前の計算をチェックポイント機能 (``checkpoint = true``) を
    有効にして実行し、最終状態のチェックポイントファイルが保存されている必要がある。

    .. note::
       ``--init`` / ``--resume`` / ``--cont`` は排他であり、同時に指定できない。
       また、実行モードへの対応状況はアルゴリズムごとに異なる。各アルゴリズムのページ (:doc:`/algorithm/index`) の「制限事項」または「リスタート」の項を参照。

**--reset_rand**
    ``--resume`` または ``--cont`` と組み合わせて使用し、再開時に乱数系列を新しくする。

**--nalg NALG**
    探索アルゴリズム層に割り当てる MPI プロセス数。
    ``--nsolve`` と組み合わせて MPI コミュニケータを分割する。両方を指定した場合は ``NALG × NSOLVE`` が総プロセス数と一致する必要がある。
    一方のみを指定した場合、もう一方は「総プロセス数 ÷ 指定値」として自動的に決定される（割り切れない場合はエラー）。

**--nsolve NSOLVE**
    ソルバーグループあたりの MPI プロセス数。
    ``--nalg`` と同様に、一方のみを指定した場合はもう一方が自動的に決定される。
    ``--nalg`` と ``--nsolve`` の両方を省略した場合は、すべてのプロセスがアルゴリズム層に割り当てられる (``NALG = 総プロセス数``, ``NSOLVE = 1``)。
    二層並列の詳細は :doc:`/tutorial/parallel_solver` を参照。

**--version**
    バージョンを表示して終了する。

**-h, --help**
    ヘルプメッセージを表示して終了する。

USAGE
-----

1. 初期状態から実行する

   .. code-block:: bash

      odatse input.toml

2. MPI 並列で実行する (4プロセス)

   .. code-block:: bash

      mpiexec -np 4 odatse input.toml

3. 中断した計算をチェックポイントから再開する

   .. code-block:: bash

      odatse --resume input.toml

4. 終了した計算を、乱数系列を新しくして延長する

   .. code-block:: bash

      odatse --cont --reset_rand input.toml

   3. と 4. は、直前の計算を ``checkpoint = true`` を指定して実行し、
   チェックポイントファイルが残っていることが前提となる。

5. 8プロセスをアルゴリズム層 2 × ソルバーグループ 4 に分割して実行する

   .. code-block:: bash

      mpiexec -np 8 odatse --nalg 2 --nsolve 4 input.toml

ENVIRONMENT
-----------

**ODATSE_NOMPI**
    ``0`` 以外の値を設定すると、mpi4py の読み込みを行わず、MPI を無効化した状態で実行する。
    mpi4py がインストールされている環境で、MPI を使わずに（mpi4py をロードせずに）実行したい場合に使用する。

.. code-block:: bash

   ODATSE_NOMPI=1 odatse input.toml

SEE ALSO
--------

- :doc:`/input/index` -- 入力ファイルの仕様
- :doc:`/output` -- 出力ファイルの仕様
- :doc:`/algorithm/index` -- 探索アルゴリズムと実行モードの対応状況
- :doc:`/tutorial/parallel_solver` -- 二層 MPI 並列の解説
