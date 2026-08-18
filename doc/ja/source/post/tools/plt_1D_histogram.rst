odatse_plt_1D_histogram
=======================

機能概要
------------
1次元周辺化ヒストグラムを作成します

書式
------------

.. code-block:: bash

   odatse_plt_1D_histogram [OPTION]... [FILE]...

説明
------------

FILE に指定するデータファイルから1次元に周辺化したヒストグラムを作成します。

データファイルはテキスト形式で、複数のカラムからなる数値データです。
標準フォーマットでは、空白文字区切りで T(または beta), fx, x1, ..., xN, weight の各数値を格納します。
T は温度 (beta=1/T は逆温度)、x1, ... xN はパラメータ値(N はパラメータの次元数)、fx はその点での関数の値、weight は重み値を表します。
フィールド名はオプション (field_list) で指定できるほか、PAMC計算に用いた入力ファイルのパラメータ (label_list) を用いることができます。

FILE を指定しない場合、オプション (data_dir) で指定したディレクトリから result_*_summarized.txt というファイル名のファイルをデータファイルとして読み込みます。

ヒストグラムを作成する軸は columns オプションで指定します。指定がない場合は x1, ..., xN のすべての軸が対象となります。指定方法はフィールド名をカンマ区切りで列挙します。例えば ``--columns x1,x3`` を指定すると ``x1`` および ``x3`` 軸に周辺化したヒストグラムを描画します。

ヒストグラムの範囲は range オプションで指定できます。その場合は表示するすべての軸について共通の range が使われます。軸ごとに指定する場合は config ファイルに ``[xmin, xmax]`` の組をリストの形で与えるか、入力パラメータファイルの ``min_list``, ``max_list`` を利用します。

.. note::
   * ODAT-SE 本体と同様に Python 3.9 以上が必要です。
   * プログレスバー表示には tqdm ライブラリが必要です。未インストールの場合は通常のメッセージが表示されます。
   * 大きなデータセットを処理する場合はメモリ使用量に注意してください。

指定可能なコマンドラインオプションを以下に示します。
これらのオプションを一括して config ファイルで与えることもできます。config ファイルは TOML 形式で、オプション名 = 値の書式でオプションを指定します。

**-b BINS, --bins BINS**
    bin の数を指定します。デフォルト値は 60 です。

**-c COLUMNS, --columns COLUMNS**
    ヒストグラムを作成するフィールド名を指定します。カンマ区切りで複数のフィールド名を指定できます。省略した場合はすべての軸が対象となります。

**-d DATA_DIR, --data_dir DATA_DIR**
    データファイルをディレクトリから取得する場合(``FILE`` を指定しない場合)のディレクトリを指定します。指定しない場合はカレントディレクトリが使われます。

**-f FORMAT, --format FORMAT**
    出力するヒストグラムファイルのフォーマットを指定します。matplotlib がサポートするフォーマットを指定可能です。カンマ区切りで複数のフォーマットを指定できます。デフォルト値は ``png`` です。

**-o OUTPUT_DIR, --output_dir OUTPUT_DIR**
    ヒストグラムファイルを出力するディレクトリを指定します。指定しない場合はカレントディレクトリに書き出されます。ディレクトリが存在しない場合は自動的に作成されます。

**-r RANGE, --range RANGE**
    ヒストグラムの範囲を xmin,xmax の形式で指定します。range コマンドラインオプションで指定した場合、すべての軸について共通になります。軸ごとに変える場合はパラメータファイルまたは config ファイルで指定します。いずれにも指定がない場合は軸ごとに自動設定されます。

**-w WEIGHT_COLUMN, --weight_column WEIGHT_COLUMN**
    weight 値のカラム番号 (0スタート) を指定します。デフォルト値は -1 (最後のカラム) です。

**--config CONFIG**
    config ファイルを指定します。config ファイルは TOML 形式で、コマンドラインオプションと同等のものを指定します。オプションの優先度はパラメータファイル < configファイル < コマンドラインオプションの順です。

**--params PARAMS**
    PAMCを実行する際に用いた入力パラメータファイルを指定します。パラメータファイルからは range (min_list, max_list) および field_list (label_list) の情報を取得します。

**--field_list FIELD_LIST**
    フィールド名を指定します。指定しない場合は標準フォーマットを仮定し、 T(or beta), fx, x1, .. xN, weight となります (Nはパラメータの次元)。パラメータファイルから取得する場合は x1 .. xN に label_list の値を用います。
    columns のフィールド名指定に使われます。

**--progress**
    実行時にプログレスバーを表示します。表示には tqdm ライブラリが必要です。tqdmがインストールされていない場合は、代わりに各ファイルの処理状況がメッセージとして表示されます。

**--xlabel XLABEL**
    x軸のラベル文字列を指定します。

**-h, --help**
    ヘルプメッセージを表示してプログラムを終了します。

使用例
------------

1. 入力データファイル file.txt を指定して実行します。出力先は 1dhist ディレクトリです。

   .. code-block:: bash

      $ odatse_plt_1D_histogram -o 1dhist file.txt

   1dhist/1Dhistogram_file.png が出力されます。

2. 入力データファイルが data ディレクトリに result_T0_summarized.txt 〜 result_T10_summarized.txt として用意されている場合の例です。出力先は 1dhist ディレクトリとします。

   .. code-block:: bash

      $ odatse_plt_1D_histogram -d data -o 1dhist

   1dhist ディレクトリに 1Dhistogram_result_T0_NNNN.png 〜 1Dhistogram_result_T10_MMMM.png が出力されます。ファイル名の ``summarized`` は ``T_{T}`` または ``beta_{beta}`` に置き換えられます。

3. 入力データ file.txt のうち、x1 と x3 のフィールドについてヒストグラムを作成し、png と pdf 形式で出力します。

   .. code-block:: bash

      $ odatse_plt_1D_histogram -c x1,x3 -o 1dhist -f png,pdf file.txt

   1dhist/1Dhistogram_file.png と 1dhist/1Dhistogram_file.pdf が出力されます。

4. 値の範囲を 3.0〜6.0 とします。すべての軸について同じ範囲に設定されます。

   .. code-block:: bash

      $ odatse_plt_1D_histogram -r 3.0,6.0 -o 1dhist file.txt

5. オプションの内容を config ファイルに記述して利用します。conf.toml を以下のように用意します。

   .. code-block:: toml

      field_list = ["beta", "fx", "z1", "z2", "z3", "weight"]
      columns = ["z1", "z2"]
      bins = 120
      range = [[3.0, 6.0], [-3.0, 3.0], [0.0, 3.0]]
      data_dir = "./summarized"
      output_dir = "1dhist"

   軸のラベルは z1, z2, z3 とし、それぞれの値の範囲はそれぞれ 3.0〜6.0, -3.0〜3.0, 0.0〜3.0 とします。
   その中で z1 と z2 についてヒストグラムを描画します。

   config ファイルを指定して実行します。

   .. code-block:: bash

      $ odatse_plt_1D_histogram --config conf.toml

   summarized/ ディレクトリ内の各 result_T*_summarized.txt についてヒストグラムが作成され、1dhist/1Dhistogram_result_T*.png に出力されます。

補足事項
------------

データファイルの形式
~~~~~~~~~~~~~~~~~~~~

標準フォーマットのデータファイルは以下の形式をとります。

.. code-block:: text

   # コメント行(任意)
   T(or beta)_value fx_value x1_value x2_value ... xN_value weight_value
   T(or beta)_value fx_value x1_value x2_value ... xN_value weight_value
   ...

各行は空白文字で区切られた数値データであり、各列は以下の意味を持ちます:

* 第1列: T(温度) または beta(逆温度)
* 第2列: fx(関数値)
* 第3列〜第(N+2)列: パラメータ値 x1, x2, ..., xN
* 最終列: 重み(weight)

ヒストグラム作成の仕組み
~~~~~~~~~~~~~~~~~~~~~~~~

このスクリプトは以下の手順でヒストグラムを作成します:

1. 入力ファイルからデータを読み込みます
2. 重みを正規化します(合計が1になるように)
3. 指定された各変数(列)に対して1次元ヒストグラムを作成します
4. 各ヒストグラムを指定されたフォーマットで保存します

出力ファイルの命名規則:

* 通常のファイル:

  ``1Dhistogram_{入力ファイル名}.{フォーマット}``

* ``odatse_summarize_each_T`` から出力された、ファイル名に _summarized.txt を含むファイル:

  ``1Dhistogram_{入力ファイル名の_summarizedを T または beta に置換}.{フォーマット}``

パフォーマンス
~~~~~~~~~~~~~~

* 大きなデータファイルを処理する場合、必要なメモリ量はファイルサイズにほぼ比例します
* NumPyを使用しているため、処理速度は比較的高速です
* 多数のファイルを処理する場合、``--progress`` オプションで進捗を確認できます

エラー処理と制限事項
~~~~~~~~~~~~~~~~~~~~

* データファイルが見つからない場合: エラーメッセージを表示します
* データ形式が不正(数値でない、列数が一致しない): そのファイルをスキップしてエラーメッセージを表示します
* フィールド名が存在しない: キーエラーが発生します
* 出力ディレクトリに書き込めない場合: 権限エラーが表示されます

処理中にエラーが発生した場合、そのファイルはスキップされて次のファイルの処理が継続されます。
最後に成功・失敗の要約が表示されます。
