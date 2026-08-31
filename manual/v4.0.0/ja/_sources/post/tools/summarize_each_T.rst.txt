odatse_summarize_each_T
=======================

機能概要
------------
PAMCの出力ファイルから各温度点での annealing 後のデータを抽出します


書式
------------

.. code-block:: bash

   odatse_summarize_each_T [OPTION]...


説明
------------

PAMCの計算において、プロセスごとの各温度点での MCMC の出力ファイル result_T*.txt から、annealing が完了した時点のレプリカのデータを抽出します。データは指定したディレクトリ内に温度点ごとのファイルとして格納します。

PAMCの計算データは DATA_DIRECTORY/[プロセス番号]/result_T[温度インデックス].txt の形式で配置されているものとします。
各ファイルの書式はスペース区切りの数値データで、MCMCステップ数(step)、レプリカ番号(walker)、温度(T)または逆温度(beta)、fx、座標値(x1 .. xN, Nは次元数)、weight、ancestor とします。

出力データは EXPORT_DIRECTORY/result_T[温度インデックス]_summarized.txt の形式で配置されます。
各ファイルの書式は、温度(T)または逆温度(beta)、fx、座標値(x1 .. xN)、weight となります。

PAMCの計算に用いた入力パラメータファイルを INPUT_FILE として指定した場合、レプリカ数(nreplica)と計算データを格納するディレクトリ(data_directory)を入力ファイルから取得します。ただし、コマンドライン引数が優先されます。

.. note::
   * ODAT-SE 本体と同様に Python 3.9 以上が必要です。
   * デフォルトでは、各ファイルの最後の nreplica 行を抽出します。この行数はレプリカ数に相当します。
   * nreplica が指定されていない場合、最後のMCMCステップを自動で判別してデータを抽出します。
   * プログレスバー表示には tqdm ライブラリが必要です。未インストールの場合はプログレスバーなしで処理が実行されます。
   * 出力ディレクトリが存在しない場合は自動的に作成されます。

指定可能なコマンドラインオプションを以下に示します。

**-i INPUT_FILE, --input_file INPUT_FILE**
    PAMCの計算に用いたTOML形式の入力パラメータファイルを指定します。指定すると、そのファイルからレプリカ数と出力ディレクトリを読み取ります。

**-n NREPLICA, --nreplica NREPLICA**
    プロセスあたりのレプリカ数を指定します。指定しない場合で、入力ファイルも指定されていない場合は、各ファイルの最後のステップのデータのみを抽出します。

**-d DATA_DIRECTORY, --data_directory DATA_DIRECTORY**
    PAMCの計算データが格納されたディレクトリを指定します。入力ファイルが指定されている場合でも、このオプションが優先されます。

**-o EXPORT_DIRECTORY, --export_directory EXPORT_DIRECTORY**
    抽出したデータを書き出すディレクトリを指定します。デフォルトは "summarized" です。

**--progress**
    実行時にプログレスバーを表示します。表示には tqdm ライブラリが必要です。

**-h, --help**
    ヘルプメッセージを表示してプログラムを終了します。

使用例
------------

1. 基本的な使用方法

   .. code-block:: bash

      odatse_summarize_each_T -d output -o summarized

   output ディレクトリ内のすべてのプロセスフォルダから result_T*.txt ファイルを処理し、summarized ディレクトリに保存します。
   各ファイルの最後のMCステップのデータが抽出されます。

2. TOML 設定ファイルを使用します

   .. code-block:: bash

      odatse_summarize_each_T -i input.toml -o summarized

   input.toml から設定を読み込み(レプリカ数、データディレクトリ)、データを処理して summarized ディレクトリに保存します。

3. レプリカ数を明示的に指定します

   .. code-block:: bash

      odatse_summarize_each_T -d output -n 16 -o summarized

   各ファイルの最後の16行を抽出します(16レプリカの場合)。

4. プログレスバーを表示します

   .. code-block:: bash

      odatse_summarize_each_T -d output -o summarized --progress

   処理中にプログレスバーを表示します(tqdmライブラリが必要)。


補足事項
------------

データ変換の詳細
~~~~~~~~~~~~~~~~

このスクリプトは以下のデータ変換を行います:

1. 入力データの形式:

   入力パラメータが温度 Tmin, Tmax で与えられた場合は

   .. code-block:: text

      step walker_id T fx x1 ... xN weight ancestor

   または、入力パラメータが逆温度 bmin, bmax で与えられた場合は

   .. code-block:: text

      step walker_id beta fx x1 ... xN weight ancestor

   各カラムの内容はファイルのヘッダ部分にコメントとして記述されます。

2. 出力データの形式:

   .. code-block:: text

      T fx x1 ... xN weight

   または、入力データが beta で与えられている場合は

   .. code-block:: text

      beta fx x1 ... xN weight

主な変換ポイント:
   * 最後のMCステップのデータの抽出
   * 不要なカラム(step、walker_id、ancestor)の削除


TOML設定ファイルの形式
~~~~~~~~~~~~~~~~~~~~~~

``-i`` で指定する TOML 設定ファイルには、PAMC 本計算を実行した際の入力ファイル (``input.toml``) をそのまま指定できます。``[base]`` セクションの ``output_dir`` と ``[algorithm.pamc]`` セクションのレプリカ数が読み込まれます。想定される形式は以下の通りです(PAMC の入力ファイルについては :doc:`../../algorithm/pamc` を参照):

.. code-block:: toml

   [base]
   output_dir = "output"  # データディレクトリ

   [algorithm.pamc]
   nreplica_per_proc = 16  # プロセスあたりのレプリカ数

必要なセクションとパラメータが設定ファイルにない場合、エラーが発生する可能性があります。

処理の仕組み
~~~~~~~~~~~~

このスクリプトは以下の手順で処理を行います:

1. コマンドライン引数を解析します(または TOML 設定ファイルから読み込みます)
2. 出力ディレクトリを作成します(存在しない場合)
3. 入力ファイルのパターンマッチングを行います(DATA_DIRECTORY/\*/result_T*.txt)
4. 各ファイルを順に処理します:

   a. ファイルを読み込みます
   b. レプリカ数が指定されている場合は最後の n 行を抽出します
   c. レプリカ数が指定されていない場合は最後のステップの行を抽出します
   d. データ変換処理を行います(不要なカラムの削除)
   e. 結果を出力ファイルに書き込みます(追記)

パフォーマンスと注意点
~~~~~~~~~~~~~~~~~~~~~~

* 一度に多数のファイルを処理する場合に ``--progress`` オプションを使用して処理の進行状況を可視化できます。
* 非常に大きなファイルを処理する場合、メモリ使用量に注意が必要です。
* 温度ごとの出力ファイルは、実行内で最初に書き込むときに上書き(truncate)し、以降の入力ファイルに対しては追記します。このため、再実行しても結果は重複せず上書きされます。
* ``pip`` で ODAT-SE をインストールした場合、Python 3.11未満で必要となる ``tomli`` は依存関係として自動的にインストールされます。

エラー処理
~~~~~~~~~~

* 入力ファイルが見つからない場合: ファイルの処理はスキップされ、エラーメッセージが表示されます。
* 出力ディレクトリに書き込み権限がない場合: 権限エラーが発生します。
* データ行のフォーマットが想定と異なる場合(カラム数不足など): 該当行の処理中にエラーが発生する可能性があります。
* TOML設定ファイルのフォーマットが正しくない場合: パース時にエラーが発生します。

スクリプトは各ファイルを try-except ブロックで処理するため、一つのファイルでエラーが発生しても他のファイルの処理は継続されます。
