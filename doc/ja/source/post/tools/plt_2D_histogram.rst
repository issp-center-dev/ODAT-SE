plt_2D_histogram.py
===================

NAME
----
2次元周辺化ヒストグラムを作成する。

SYNOPSIS
--------

.. code-block:: bash

   python3 plt_2D_histogram.py [OPTION]... [FILE]...

DESCRIPTION
-----------

FILE に指定するデータファイルから2次元に周辺化したヒストグラムを作成する。

データファイルはテキスト形式で、複数のカラムからなる数値データである。
標準フォーマットでは、空白文字区切りで beta, fx, x1, ..., xN, weight の各数値を格納する。
beta は逆温度、x1, ... xN はパラメータ値(N はパラメータの次元数)、fx はその点での関数の値、weight は重み値を表す。
フィールド名はオプション (field_list) で指定できるほか、PAMC計算に用いた入力ファイルのパラメータ (label_list) を用いることができる。

FILE を指定しない場合、オプション (data_dir) で指定したディレクトリから result_*_summarized.txt というファイル名のファイルをデータファイルとして読み込む。

ヒストグラムを作成する軸は columns オプションで指定する。指定した軸のすべての組み合わせの2次元プロットが作成される。指定がない場合は x1, ..., xN のすべての軸が対象となる。指定方法はフィールド名をカンマ区切りで列挙する。例えば ``--column x1,x2,x3`` を指定すると ``x1 vs x2`` 軸、 ``x1 vs x3`` 軸、および ``x2 vs x3`` 軸に周辺化したヒストグラムを描画する。

ヒストグラムの範囲は range オプションで指定できる。その場合は表示するすべての軸について共通の range が使われる。軸ごとに指定する場合は config ファイルに ``[xmin, xmax]`` の組をリストの形で与えるか、入力パラメータファイルの ``min_list``, ``max_list`` を利用する。

指定可能なコマンドラインオプションを以下に示す。
これらのオプションを一括して config ファイルで与えることもできる。config ファイルは TOML 形式で、オプション名 = 値の書式でオプションを指定する。

**-b BINS, --bins BINS**
    bin の数を指定する。
    
**-c COLUMNS, --columns COLUMNS**
    ヒストグラムを作成するフィールド名を指定する。カンマ区切りで複数のフィールド名を指定できる。省略した場合はすべての軸が対象となる。
			
**-d DATA_DIR, --data_dir DATA_DIR**
    データファイルをディレクトリから取得する場合(``file`` を指定しない場合)のディレクトリを指定する。指定しない場合はカレントディレクトリが使われる。
			
**-f FORMAT, --format FORMAT**
    出力するヒストグラムファイルのフォーマットを指定する。matplotlib がサポートするフォーマットを指定可能。カンマ区切りで複数のフォーマットを指定できる。デフォルト値は ``png`` 。

**-o OUTPUT_DIR, --output_dir OUTPUT_DIR**
    ヒストグラムファイルを出力するディレクトリを指定する。指定しない場合はカレントディレクトリに書き出される。

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
    実行時にプログレスバーを表示する。表示には tqdm ライブラリが必要。
    
**--xlabel XLABEL**
    x軸のラベル文字列を指定する。
    
**-h, --help**
    ヘルプメッセージを表示してプログラムを終了する。

USAGE
-----

1. 入力データファイル file.txt を指定して実行する。出力先は 2dhist ディレクトリ。

   .. code-block:: bash

      $ python3 plt_2D_histogram.py -o 2dhist file.txt

   2dhist/2Dhistogram_file_x1_vs_x2.png,
   2dhist/2Dhistogram_file_x1_vs_x3.png,
   2dhist/2Dhistogram_file_x2_vs_x3.png が出力される。

2. 入力データファイルが data ディレクトリに result_T0_summarized.txt 〜 result_T10_summarized.txt として用意されている場合。出力先は 2dhist ディレクトリとする。

   .. code-block:: bash

      $ python3 plt_2D_histogram.py -d data -o 2dhist

   2dhist ディレクトリに 2Dhistogram_result_T0_beta_{beta}_x1_vs_x2.png 〜 2Dhistogram_result_T10_beta_{beta}_x2_vs_x3.png が出力される。ファイル名の ``summarized`` は ``beta_{beta}`` に置き換えられる。

3. 入力データ file.txt のうち、x1, x3 のフィールドについて2次元ヒストグラムを作成し、png と pdf 形式で出力する。

   .. code-block:: bash

      $ python3 plt_2D_histogram.py -c x1,x3 -o 2dhist -f png,pdf file.txt

   2dhist/2Dhistogram_file_x1_vs_x3.png と 2dhist/2Dhistogram_file_x1_vs_x3.pdf が出力される。

4. 値の範囲を 3.0〜6.0 とする。すべての軸について同じ範囲に設定される。

   .. code-block:: bash

      $ python3 plt_2D_histogram.py -r 3.0,6.0 -o 2dhist file.txt

5. オプションの内容を config ファイルに記述して利用する。conf.toml を以下のように用意する。

   .. code-block:: toml

      field_list = ["beta", "fx", "z1", "z2", "z3", "weight"]
      columns = ["z1", "z2"]
      bins = 120
      range = [[3.0, 6.0], [-3.0, 3.0], [0.0, 3.0]]
      data_dir = "./summarized"
      output_dir = "2dhist"

   軸のラベルは z1, z2, z3 とし、それぞれの値の範囲はそれぞれ 3.0〜6.0, -3.0〜3.0, 0.0〜3.0 とする。
   その中で z1 vs z2 についてヒストグラムを描画する。

   config ファイルを指定して実行する。

   .. code-block:: bash

      $ python3 plt_2D_histogram.py --config config.toml

   summarized/ ディレクトリ内の各 result_T*_summarized.txt についてヒストグラムが作成され、2dhist/2Dhistogram_result_T*.png に出力される。
