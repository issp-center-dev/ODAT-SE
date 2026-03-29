========================================
関連リソース
========================================

ODAT-SE Gallery
~~~~~~~~~~~~~~~~~~~~~~~~~

`ODAT-SE Gallery <https://isspns-gitlab.issp.u-tokyo.ac.jp/takeohoshi/odat-se-gallery>`_ は、ODAT-SE を用いた各種解析手法のサンプルデータ、実行例、ソルバーテンプレートを集めたリポジトリです。

解析例
^^^^^^^^^^^^^^^^^^^^^^^^

以下の量子ビーム回折実験に対する解析例が含まれています。

- **odatse-STR** -- 全反射高速陽電子回折 (TRHEPD) の構造解析例
- **odatse-SXRD** -- 表面X線回折 (SXRD) の解析例
- **odatse-LEED** -- 低速電子線回折 (LEED) の解析例
- **odatse-XAFS** -- 広域X線吸収微細構造 (XAFS) の解析例

各サンプルにはメッシュデータ、入力ファイル (``input.toml``)、実行スクリプト (``do.sh``)、参照データ、可視化スクリプトが含まれており、
そのまま実行して結果を確認できます。

ソルバーテンプレート
^^^^^^^^^^^^^^^^^^^^^^^^

独自のソルバーを開発するためのテンプレートが4種類用意されています。
詳細は :doc:`customize/tutorial_solver` を参照してください。

- **user_function** -- Python 関数を手軽に最適化するための最小構成スクリプト
- **function_module** -- 解析関数ソルバーの ``pip install`` 可能なパッケージテンプレート
- **solver_module** -- 参照データとの比較を行うソルバーのテンプレート
- **external_solver_module** -- 外部プログラム (C/Fortran 等) をソルバーとして利用するためのテンプレート

入手方法
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ git clone https://isspns-gitlab.issp.u-tokyo.ac.jp/takeohoshi/odat-se-gallery.git


関連リンク
~~~~~~~~~~~~~~~~~~~~~~~~~

- `ODAT-SE GitHub リポジトリ <https://github.com/issp-center-dev/ODAT-SE>`_
- `ソフトウェア開発・高度化プロジェクト <https://www.pasums.issp.u-tokyo.ac.jp/>`_
- `ODAT-SE 関連発表資料 <https://www.pasums.issp.u-tokyo.ac.jp/odat-se/paper>`_
