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

指定可能なコマンドラインオプションを以下に示す。
FILE を指定した場合はそのファイルが対象となる。明示的にファイルを指定しない場合、DATA_DIR/\*/FILE_TYPE が対象となる。

**FILE**
    MCMCのログファイルを指定する。複数のファイルを指定可能。

**-d DATA_DIR, --data_dir DATA_DIR**
    データファイルをディレクトリから取得する場合(``FILE`` を指定しない場合)に、ディレクトリを指定する。
			
**-t FILE_TYPE, --file_type FILE_TYPE**
    ディレクトリを指定して実行する場合に、対象となるファイル名を指定する。デフォルトは result.txt 。

**--progress**
    実行時にプログレスバーを表示する。表示には tqdm ライブラリが必要。

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
