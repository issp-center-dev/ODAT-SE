summarize_each_T.py
===================

NAME
----
PAMCの出力ファイルから各温度点での annealing 後のデータを抽出する


SYNOPSIS
--------

.. code-block:: bash

   python3 summarize_each_T.py [OPTION]...


DESCRIPTION
-----------

PAMCの計算において、プロセスごとの各温度点での MCMC の出力ファイル result_T*.txt から、annealing が完了した時点のレプリカのデータを抽出する。データは指定したディレクトリ内に温度点ごとのファイルとして格納する。

PAMCの計算データは DATA_DIRECTORY/[プロセス番号]/result_T[温度インデックス].txt の形式で配置されているものとする。
各ファイルの書式はスペース区切りの数値データで、MCMCステップ数(step)、レプリカ番号(walker)、温度(T)、fx、座標値(x1 .. xN, Nは次元数)、weight、ancestor とする。

出力データは EXPORT_DIRECTORY/result_T[温度インデックス]_summarized.txt の形式で配置される。
各ファイルの書式は、逆温度(beta)、fx、座標値(x1 .. xN)、weight となる。

PAMCの計算に用いた入力パラメータファイルを INPUT_FILE として指定した場合、レプリカ数(nreplica)と計算データを格納するディレクトリ(data_directory)を入力ファイルから取得する。ただし、コマンドライン引数が優先される。

.. note::
   * Python 3.6以上が必要です(タイプヒントとf文字列を使用しているため)。
   * 逆温度(beta)は温度(T)の逆数として計算されます(beta = 1/T)。T = 0 の場合は beta = 0 とされます。
   * デフォルトでは、各ファイルの最後の nreplica 行を抽出します。この行数はレプリカ数に相当します。
   * nreplica が指定されていない場合、最後のMCMCステップを自動で判別してデータを抽出します。
   * プログレスバー表示には tqdm ライブラリが必要です。未インストールの場合はプログレスバーなしで処理が実行されます。
   * 出力ディレクトリが存在しない場合は自動的に作成されます。

指定可能なコマンドラインオプションを以下に示す。

**-i INPUT_FILE, --input_file INPUT_FILE**
    PAMCの計算に用いたTOML形式の入力パラメータファイルを指定する。指定すると、そのファイルからレプリカ数と出力ディレクトリを読み取る。

**-n NREPLICA, --nreplica NREPLICA**
    プロセスあたりのレプリカ数を指定する。指定しない場合で、入力ファイルも指定されていない場合は、各ファイルの最後のステップのデータのみを抽出する。

**-d DATA_DIRECTORY, --data_directory DATA_DIRECTORY**
    PAMCの計算データが格納されたディレクトリ。入力ファイルが指定されている場合でも、このオプションが優先される。

**-o EXPORT_DIRECTORY, --export_directory EXPORT_DIRECTORY**
    抽出したデータを書き出すディレクトリ。デフォルトは "summarized"。

**--progress**
    実行時にプログレスバーを表示する。表示には tqdm ライブラリが必要。

**-h, --help**
    ヘルプメッセージを表示してプログラムを終了する。

USAGE
-----

1. 基本的な使用方法

   .. code-block:: bash

      python3 summarize_each_T.py -d output -o summarized

   output ディレクトリ内のすべてのプロセスフォルダから result_T*.txt ファイルを処理し、summarized ディレクトリに保存します。
   各ファイルの最後のMCステップのデータが抽出されます。

2. TOML 設定ファイルを使用する

   .. code-block:: bash

      python3 summarize_each_T.py -i input.toml -o summarized

   input.toml から設定を読み込み(レプリカ数、データディレクトリ)、データを処理して summarized ディレクトリに保存します。

3. レプリカ数を明示的に指定する

   .. code-block:: bash

      python3 summarize_each_T.py -d output -n 16 -o summarized

   各ファイルの最後の16行を抽出します(16レプリカの場合)。

4. プログレスバーを表示する

   .. code-block:: bash

      python3 summarize_each_T.py -d output -o summarized --progress

   処理中にプログレスバーを表示します(tqdmライブラリが必要)。


NOTES
-----

データ変換の詳細
~~~~~~~~~~~~~~~~

このスクリプトは以下のデータ変換を行います：

1. 入力データの形式:

   .. code-block:: text

      step walker_id T fx x1 ... xN weight ancestor

2. 出力データの形式:

   .. code-block:: text

      beta fx x1 ... xN weight

主な変換ポイント:
   * 最後のMCステップのデータの抽出
   * 温度(T)から逆温度(beta = 1/T)への変換
   * 不要なカラム(step、walker_id、ancestor)の削除

温度(T)が 0 の場合は、逆温度(beta)も 0 として設定されます。

TOML設定ファイルの形式
~~~~~~~~~~~~~~~~~~~~~~

指定するTOML設定ファイルは以下の形式が想定されています:

.. code-block:: toml

   [base]
   output_dir = "output"  # データディレクトリ

   [algorithm.pamc]
   nreplica_per_proc = 16  # プロセスあたりのレプリカ数

必要なセクションとパラメータが設定ファイルにない場合、エラーが発生する可能性があります。

処理の仕組み
~~~~~~~~~~~~

このスクリプトは以下の手順で処理を行います:

1. コマンドライン引数の解析(または TOML 設定ファイルからの読み込み)
2. 出力ディレクトリの作成(存在しない場合)
3. 入力ファイルのパターンマッチング(DATA_DIRECTORY/\*/result_T*.txt)
4. 各ファイルの処理:
   
   a. ファイルを行単位で読み込み(コメント行を除外)
   b. レプリカ数が指定されている場合は最後の n 行を抽出
   c. レプリカ数が指定されていない場合は最後のステップの行を抽出
   d. データ変換処理(温度→逆温度、不要なカラムの削除)
   e. 結果を出力ファイルに書き込み

パフォーマンスと注意点
~~~~~~~~~~~~~~~~~~~~~~

* 一度に多数のファイルを処理する場合に `--progress` オプションを使用して処理の進行状況を可視化できます。
* 非常に大きなファイルを処理する場合、メモリ使用量に注意が必要です。
* 出力ファイルに追記モード (`a`) で書き込むため、同じ処理を複数回実行すると結果が重複する可能性があります。再実行する場合は、出力ディレクトリを空にするか新しいディレクトリを指定してください。
* TOMLファイルから設定を読み込む場合、Python 3.11未満では追加のライブラリ(tomli)が必要です。

エラー処理
~~~~~~~~~~

* 入力ファイルが見つからない場合: ファイルの処理はスキップされ、エラーメッセージが表示されます。
* 出力ディレクトリに書き込み権限がない場合: 権限エラーが発生します。
* データ行のフォーマットが想定と異なる場合(カラム数不足など): 該当行の処理中にエラーが発生する可能性があります。
* TOML設定ファイルのフォーマットが正しくない場合: パース時にエラーが発生します。

スクリプトは各ファイルを try-except ブロックで処理するため、一つのファイルでエラーが発生しても他のファイルの処理は継続されます。
