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

指定可能なコマンドラインオプションを以下に示す。

**-i INPUT_FILE, --input_file INPUT_FILE**
    PAMCの計算に用いたTOML形式の入力パラメータファイルを指定する。

**-n NREPLICA, --nreplica NREPLICA**
    プロセスあたりのレプリカ数を指定する。指定しない場合はデータファイルから自動で判別する。

**-d DATA_DIRECTORY, --data_directory DATA_DIRECTORY**
    PAMCの計算データを格納するディレクトリ。

**-o EXPORT_DIRECTORY, --export_directory EXPORT_DIRECTORY**
    抽出したデータを書き出すディレクトリ。

**--progress**
    実行時にプログレスバーを表示する。表示には tqdm ライブラリが必要。

**-h, --help**
    ヘルプメッセージを表示してプログラムを終了する。

