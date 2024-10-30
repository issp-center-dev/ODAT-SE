py2dmat のインストール
=============================

実行環境・必要なパッケージ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- python 3.6.8 以上

    - 必要なpythonパッケージ

        - tomli (>= 1.2)
        - numpy (>= 1.14)

    - Optional なパッケージ

        - mpi4py (グリッド探索利用時)
        - scipy (Nelder-Mead法利用時)
        - physbo (ベイズ最適化利用時, ver. 0.3以上)

ダウンロード・インストール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

下記に示す方法で、 ``py2dmat`` python パッケージと ``py2dmat`` コマンドがインストールできます。

- PyPI からのインストール（推奨）

    - ``python3 -m pip install py2dmat``

        - ``--user`` オプションをつけるとローカル (``$HOME/.local``) にインストールできます

        - ``py2dmat[all]`` とすると Optional なパッケージも同時にインストールします

- ソースコードからのインストール

    1. ``git clone https://github.com/issp-center-dev/2DMAT``
    2. ``python3 -m pip install ./2DMAT``

        - ``pip`` のバージョンは 19 以上が必要です (``python3 -m pip install -U pip`` で更新可能)

- サンプルファイルのダウンロード

    - サンプルファイルはソースコードに同梱されています。
    - ``git clone https://github.com/issp-center-dev/2DMAT``

なお、 ``py2dmat`` で用いる順問題ソルバーのうち、

- TRHEPD順問題ソルバー (``sim-trhepd-rheed``)
- SXRD順問題ソルバー (``sxrdcalc``)
- LEED順問題ソルバー (``satleed``)

については、別途インストールが必要です。
インストールの詳細については各ソルバーのチュートリアルを参照してください。


実行方法
~~~~~~~~~~~~~

``py2dmat`` コマンドは定義済みの最適化アルゴリズム ``Algorithm`` と順問題ソルバー ``Solver`` の組み合わせで解析を行います。::
    
    $ py2dmat input.toml

定義済みの ``Algorithm`` については :doc:`algorithm/index` を、
``Solver`` については :doc:`solver/index` を参照してください。

``Algorithm`` や ``Solver`` をユーザーが準備する場合は、 ``py2dmat`` パッケージを利用します。
詳しくは :doc:`customize/index` を参照してください。

なお、 プログラムを改造する場合など、 ``py2dmat`` コマンドをインストールせずに直接実行することも可能です。
``src/py2dmat_main.py`` を利用してください。::

    $ py2dmat src/py2dmat_main.py input.toml

アンインストール
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    $ python3 -m pip uninstall py2dmat
