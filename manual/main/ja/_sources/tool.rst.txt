関連ツール
=====================

``odatse_neighborlist``
*****************************

離散空間をモンテカルロ探索する場合に使用する近傍リスト定義ファイルを
メッシュ定義ファイルから生成するツールです。

``pip`` でインストールした場合は ``odatse`` と同様に ``bin`` 以下に ``odatse_neighborlist`` という名前でインストールされます。
もしくは、ディレクトリ中の ``src/odatse_neighborlist.py`` を直接実行することも可能です。

使い方
~~~~~~~~~~~~~~~~

引数としてメッシュ定義ファイルを渡します。
生成される近傍リスト定義ファイルの名前は ``-o`` オプションで指定可能です。

.. code-block:: bash

    $ odatse_neighborlist -o neighborlist.txt MeshData.txt


次のオプションが利用できます。

- ``-o output`` or ``--output output``

  - 出力ファイル名 (default: ``neighborlist.txt``)

- ``-u "unit1 unit2..."`` or ``--unit "unit1 unit2"``

  - 各次元の長さスケール (default: すべて 1.0)

    - 空間次元の数だけ値を空白区切りで並べ、全体を引用符でくくってください

  - 各座標はあらかじめこれらの長さスケールで除算されます

- ``-r radius`` or ``--radius radius``

  - 近傍とみなされるユークリッド距離 (default: 1.0)
  - 距離は ``-u`` で除されたあとの座標に対して計算されます

- ``-q`` or ``--quiet``

  - 進捗バーを表示しません
  - なお、進捗バーの表示には ``tqdm`` python パッケージが必要です

- ``--allow-selfloop``

  - 自分自身を隣接リストに含めます（自己ループ）

- ``--check-allpairs``

  - すべての点対に対して距離を計算します
  - デバッグ用のオプションです


なお、 MPI を用いて計算を高速化できます。

