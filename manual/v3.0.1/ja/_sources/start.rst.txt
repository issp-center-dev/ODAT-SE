ODAT-SE のインストール
================================

実行環境・必要なパッケージ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- python 3.9 以上

    - 必要なpythonパッケージ

        - tomli (>= 1.2)
        - numpy (>= 1.14)

    - Optional なパッケージ

        - mpi4py (グリッド探索利用時)
        - scipy (Nelder-Mead法利用時)
        - physbo (ベイズ最適化利用時, ver. 2.0以上)

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


実行方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``odatse`` コマンドは定義済みの最適化アルゴリズム ``Algorithm`` と順問題ソルバー ``Solver`` の組み合わせで解析を行います。

.. code-block:: bash
    
    $ odatse input.toml

定義済みの ``Algorithm`` については :doc:`algorithm/index` を、
``Solver`` については :doc:`solver/index` を参照してください。

2次元物質構造解析向け実験データ解析の順問題ソルバーは独立なパッケージとして提供されています。
これらの解析を行う場合は別途パッケージと必要なソフトウェアをインストールしてください。
現在は以下の順問題ソルバーが用意されています。

- 全反射高速陽電子回折 (TRHEPD) -- odatse-STR パッケージ

- 表面X線回折 (SXRD) -- odatse-SXRD パッケージ

- 低速電子線回折 (LEED) -- odatse-LEED パッケージ

``Algorithm`` や ``Solver`` をユーザーが準備する場合は、 ``ODAT-SE`` パッケージを利用します。
詳しくは :doc:`customize/index` を参照してください。

なお、 プログラムを改造する場合など、 ``odatse`` コマンドをインストールせずに直接実行することも可能です。
``src/odatse_main.py`` を利用してください。

.. code-block::

    $ python3 src/odatse_main.py input.toml


アンインストール
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ODAT-SE モジュールをアンインストールするには、以下のコマンドを実行します。

.. code-block:: bash

    $ python3 -m pip uninstall ODAT-SE
