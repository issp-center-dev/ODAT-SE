separateT.py
============

NAME
----
MCMCのログファイルを温度点ごとに分割する

SYNOPSIS
--------

.. code-block:: bash

   python3 separateT.py [OPTION]... [FILE]...


DESCRIPTION
-----------

MCMC のログファイル (result.txt, trial.txt) を温度点ごとに個別のファイルに分割する。
ファイルは入力と同じディレクトリ内に作成され、ファイル名は元のファイルに ``_T{index}`` を付けた形式になる。``{index}`` は異なる温度点のインデックスで、ログファイル内の出現順に 0 から付与される。

.. note::
   * Python 3.6以上が必要です(タイプヒントとf文字列を使用しているため)。
   * MCMCログファイルは温度値が3番目のカラム(インデックス2)にあることを前提としています。
   * 入力ファイル内のコメント行(#で始まる行)はすべての出力ファイルに保持されます。
   * プログレスバー表示には tqdm ライブラリが必要です。未インストールの場合は通常のメッセージが表示されます。

指定可能なコマンドラインオプションを以下に示す。
FILE を指定した場合はそのファイルが対象となる。明示的にファイルを指定しない場合、DATA_DIR/\*/FILE_TYPE が対象となる。

**FILE**
    MCMCのログファイルを指定する。複数のファイルを指定可能。

**-d DATA_DIR, --data_dir DATA_DIR**
    データファイルをディレクトリから取得する場合(``FILE`` を指定しない場合)に、ディレクトリを指定する。
			
**-t FILE_TYPE, --file_type FILE_TYPE**
    ディレクトリを指定して実行する場合に、対象となるファイル名を指定する。デフォルトは result.txt 。

**--progress**
    実行時にプログレスバーを表示する。表示には tqdm ライブラリが必要。tqdmがインストールされていない場合は、代わりに処理中のファイル名がメッセージとして表示される。

**-h, --help**
    ヘルプメッセージを表示してプログラムを終了する。

USAGE
-----

1. ファイル名を指定して実行する

   .. code-block:: bash

      python3 separateT.py output/0/result.txt

   output/0/result_T0.txt, output/0/result_T1.txt, ... が作成される。

2. 指定したディレクトリ以下のファイルを分割する

   .. code-block:: bash

      python3 separateT.py -d output

   output/0/result.txt, output/1/result.txt, ... が分割の対象となる。

3. 指定したディレクトリ以下の result.txt 以外のファイルを分割する。

   .. code-block:: bash

      python3 separateT.py -t trial.txt -d output

   output/0/trial.txt, output/1/trial.txt, ... を分割する。

4. 複数のファイルを一度に処理し、進捗状況を表示する。

   .. code-block:: bash

      python3 separateT.py --progress file1.txt file2.txt file3.txt

   各ファイルが温度点ごとに分割され、処理の進捗状況がプログレスバーで表示される。

NOTES
-----

ファイル形式
~~~~~~~~~~~~

入力ファイル(MCMCログファイル)は以下のような形式を想定しています:

.. code-block:: text

   # コメント行(任意)
   step replica_id T fx x1 ... xN ...
   step replica_id T fx x1 ... xN ...
   ...

各行は空白文字で区切られたデータで、3番目のカラム(インデックス2)が温度値Tです。
同じ温度値を持つ連続した行が1つのファイルにまとめられます。

処理の仕組み
~~~~~~~~~~~~

このスクリプトは以下の手順で処理を行います:

1. 入力ファイルを1行ずつ読み込む
2. コメント行(#で始まる行)をヘッダーとして記録
3. 各データ行の3番目のカラム(インデックス2)から温度値を取得
4. 温度値が変わるたびに、それまでのデータを別ファイルに書き出す
5. 各温度値のデータは、元のファイル名に「_T{インデックス}」を付けたファイルに保存

出力ファイルの形式
~~~~~~~~~~~~~~~~~~

出力ファイルは以下の形式になります:

* ファイル名: 元のファイル名に「_T{インデックス}」を追加(例: result.txt → result_T0.txt, result_T1.txt, ...)
* ファイル内容: 入力ファイルのヘッダー(コメント行)に続いて、同じ温度値を持つデータ行

パフォーマンス
~~~~~~~~~~~~~~

* ファイルを1行ずつ処理するため、非常に大きなファイルでもメモリ使用量は抑えられます
* 各温度点のデータはメモリ上にバッファリングされるため、1つの温度点に非常に多くのデータがある場合はメモリ使用量が増加する可能性があります
* 処理時間は入力ファイルのサイズとともに増加しますが、行単位の処理のため比較的高速です
* 複数のファイルを処理する場合、`--progress` オプションを使用することで進捗状況を確認できます

エラー処理
~~~~~~~~~~

* 入力ファイルが見つからない場合: ファイルオープンエラーが発生し、その旨のメッセージが表示されます
* 出力ファイルが書き込めない場合: 権限エラーなどが発生し、その旨のメッセージが表示されます
* データ行の列数が足りない場合: インデックスエラーが発生する可能性があります(3番目のカラムが存在しない場合)
