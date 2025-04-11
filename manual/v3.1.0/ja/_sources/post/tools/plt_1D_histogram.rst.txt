plt_1D_histogram.py
====================

NAME
----
1次元周辺化ヒストグラムを作成する

SYNOPSIS
--------

.. code-block:: bash

   python3 plt_1D_histogram.py [OPTION]... [FILE]...

DESCRIPTION
-----------

FILE に指定するデータファイルから1次元に周辺化したヒストグラムを作成する。

データファイルはテキスト形式で、複数のカラムからなる数値データである。
標準フォーマットでは、空白文字区切りで beta, fx, x1, ..., xN, weight の各数値を格納する。
beta は逆温度、x1, ... xN はパラメータ値(N はパラメータの次元数)、fx はその点での関数の値、weight は重み値を表す。
フィールド名はオプション (field_list) で指定できるほか、PAMC計算に用いた入力ファイルのパラメータ (label_list) を用いることができる。

FILE を指定しない場合、オプション (data_dir) で指定したディレクトリから result_*_summarized.txt というファイル名のファイルをデータファイルとして読み込む。

ヒストグラムを作成する軸は columns オプションで指定する。指定がない場合は x1, ..., xN のすべての軸が対象となる。指定方法はフィールド名をカンマ区切りで列挙する。例えば ``--column x1,x3`` を指定すると ``x1`` および ``x3`` 軸に周辺化したヒストグラムを描画する。

ヒストグラムの範囲は range オプションで指定できる。その場合は表示するすべての軸について共通の range が使われる。軸ごとに指定する場合は config ファイルに ``[xmin, xmax]`` の組をリストの形で与えるか、入力パラメータファイルの ``min_list``, ``max_list`` を利用する。

.. note::
   * Python 3.6以上が必要です(f文字列を使用しているため)。
   * プログレスバー表示には tqdm ライブラリが必要です。未インストールの場合は通常のメッセージが表示されます。
   * 大きなデータセットを処理する場合はメモリ使用量に注意してください。

指定可能なコマンドラインオプションを以下に示す。
これらのオプションを一括して config ファイルで与えることもできる。config ファイルは TOML 形式で、オプション名 = 値の書式でオプションを指定する。

**-b BINS, --bins BINS**
    bin の数を指定する。デフォルト値は 60。

**-c COLUMNS, --columns COLUMNS**
    ヒストグラムを作成するフィールド名を指定する。カンマ区切りで複数のフィールド名を指定できる。省略した場合はすべての軸が対象となる。
			
**-d DATA_DIR, --data_dir DATA_DIR**
    データファイルをディレクトリから取得する場合(``FILE`` を指定しない場合)のディレクトリを指定する。指定しない場合はカレントディレクトリが使われる。
			
**-f FORMAT, --format FORMAT**
    出力するヒストグラムファイルのフォーマットを指定する。matplotlib がサポートするフォーマットを指定可能。カンマ区切りで複数のフォーマットを指定できる。デフォルト値は ``png`` 。

**-o OUTPUT_DIR, --output_dir OUTPUT_DIR**
    ヒストグラムファイルを出力するディレクトリを指定する。指定しない場合はカレントディレクトリに書き出される。ディレクトリが存在しない場合は自動的に作成される。

**-r RANGE, --range RANGE**
    ヒストグラムの範囲を xmin,xmax の形式で指定する。range コマンドラインオプションで指定した場合、すべての軸について共通になる。軸ごとに変える場合はパラメータファイルまたは config ファイルで指定する。いずれにも指定がない場合は軸ごとに自動設定される。

**-w WEIGHT_COLUMN, --weight_column WEIGHT_COLUMN**
    weight 値のカラム番号 (0スタート) を指定する。デフォルト値は -1 (最後のカラム)。

**--config CONFIG**
    config ファイルを指定する。config ファイルは TOML 形式で、コマンドラインオプションと同等のものを指定する。オプションの優先度はパラメータファイル < configファイル < コマンドラインオプションの順。

**--params PARAMS**
    PAMCを実行する際に用いた入力パラメータファイルを指定する。パラメータファイルからは range (min_list, max_list) および field_list (label_list) の情報を取得する。

**--field_list FIELD_LIST**
    フィールド名を指定する。指定しない場合は標準フォーマットを仮定し、 beta, fx, x1, .. xN, weight となる (Nはパラメータの次元)。パラメータファイルから取得する場合は x1 .. xN に label_list の値を用いる。
    columns のフィールド名指定に使われる。

**--progress**
    実行時にプログレスバーを表示する。表示には tqdm ライブラリが必要。tqdmがインストールされていない場合は、代わりに各ファイルの処理状況がメッセージとして表示される。

**--xlabel XLABEL**
    x軸のラベル文字列を指定する。

**-h, --help**
    ヘルプメッセージを表示してプログラムを終了する。

USAGE
-----

1. 入力データファイル file.txt を指定して実行する。出力先は 1dhist ディレクトリ。

   .. code-block:: bash

      $ python3 plt_1D_histogram.py -o 1dhist file.txt

   1dhist/1Dhistogram_file.png が出力される。

2. 入力データファイルが data ディレクトリに result_T0_summarized.txt 〜 result_T10_summarized.txt として用意されている場合。出力先は 1dhist ディレクトリとする。

   .. code-block:: bash

      $ python3 plt_1D_histogram.py -d data -o 1dhist

   1dhist ディレクトリに 1Dhistogram_result_T0_beta_NNNN.png 〜 1Dhistogram_result_T10_beta_MMMM.png が出力される。ファイル名の ``summarized`` は ``beta_{beta}`` に置き換えられる。

3. 入力データ file.txt のうち、x1 と x3 のフィールドについてヒストグラムを作成し、png と pdf 形式で出力する。

   .. code-block:: bash

      $ python3 plt_1D_histogram.py -c x1,x3 -o 1dhist -f png,pdf file.txt

   1dhist/1Dhistogram_file.png と 1dhist/1Dhistogram_file.pdf が出力される。

4. 値の範囲を 3.0〜6.0 とする。すべての軸について同じ範囲に設定される。

   .. code-block:: bash

      $ python3 plt_1D_histogram.py -r 3.0,6.0 -o 1dhist file.txt

5. オプションの内容を config ファイルに記述して利用する。conf.toml を以下のように用意する。

   .. code-block:: toml

      field_list = ["beta", "fx", "z1", "z2", "z3", "weight"]
      columns = ["z1", "z2"]
      bins = 120
      range = [[3.0, 6.0], [-3.0, 3.0], [0.0, 3.0]]
      data_dir = "./summarized"
      output_dir = "1dhist"

   軸のラベルは z1, z2, z3 とし、それぞれの値の範囲はそれぞれ 3.0〜6.0, -3.0〜3.0, 0.0〜3.0 とする。
   その中で z1 と z2 についてヒストグラムを描画する。

   config ファイルを指定して実行する。

   .. code-block:: bash

      $ python3 plt_1D_histogram.py --config conf.toml

   summarized/ ディレクトリ内の各 result_T*_summarized.txt についてヒストグラムが作成され、1dhist/1Dhistogram_result_T*.png に出力される。

NOTES
-----

データファイルの形式
~~~~~~~~~~~~~~~~~~~~

標準フォーマットのデータファイルは以下の形式をとる。

.. code-block:: text

   # コメント行(任意)
   beta_value fx_value x1_value x2_value ... xN_value weight_value
   beta_value fx_value x1_value x2_value ... xN_value weight_value
   ...

各行は空白文字で区切られた数値データであり、各列は以下の意味を持つ:

* 第1列: beta値(逆温度)
* 第2列: fx値(関数値)
* 第3列〜第(N+2)列: パラメータ値 x1, x2, ..., xN
* 最終列: 重み(weight)

ヒストグラム作成の仕組み
~~~~~~~~~~~~~~~~~~~~~~~~

このスクリプトは以下の手順でヒストグラムを作成する:

1. 入力ファイルからデータを読み込む
2. 重みを正規化する(合計が1になるように)
3. 指定された各変数(列)に対して1次元ヒストグラムを作成
4. 各ヒストグラムを指定されたフォーマットで保存

出力ファイルの命名規則:

* 通常のファイル:

  ``1Dhistogram_{入力ファイル名}.{フォーマット}``

* ``summarize_each_T.py`` から出力された、ファイル名に _summarized.txt を含むファイル:

  ``1Dhistogram_{入力ファイル名の_summarizedを_beta_{beta値}に置換}.{フォーマット}``

パフォーマンス
~~~~~~~~~~~~~~

* 大きなデータファイルを処理する場合、必要なメモリ量はファイルサイズにほぼ比例する
* NumPyを使用しているため、処理速度は比較的高速
* 多数のファイルを処理する場合、``--progress`` オプションで進捗を確認できる

エラー処理と制限事項
~~~~~~~~~~~~~~~~~~~~

* データファイルが見つからない場合: エラーメッセージを表示
* データ形式が不正(数値でない、列数が一致しない): そのファイルをスキップしてエラーメッセージを表示
* フィールド名が存在しない: キーエラーが発生
* 出力ディレクトリに書き込めない場合: 権限エラーが表示される

処理中にエラーが発生した場合、そのファイルはスキップされて次のファイルの処理が継続される。
最後に成功・失敗の要約が表示される。
