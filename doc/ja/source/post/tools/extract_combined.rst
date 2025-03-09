extract_combined.py
===================

NAME
----
combined 形式のログファイルから特定の項目を取り出す

SYNOPSIS
--------

.. code-block:: bash

   python3 extract_combined.py [OPTION]... -t tag [FILE]...


DESCRIPTION
-----------

MCMC の combined 形式のログファイルから tag でタグ付けされた行を取り出す。

combined 形式では、行の先頭に ``"<tag> "`` 形式のタグを付けて複数のデータが格納されている。このファイルから特定の ``tag`` のついた行を取り出し、ファイルに出力する。出力ファイルは入力と同じディレクトリ内の ``tag`` というファイルになる。

指定可能なコマンドラインオプションを以下に示す。
FILE を指定した場合はそのファイルが対象となる。明示的にファイルを指定しない場合、DATA_DIR/\*/combined.txt が対象となる。

**FILE**
    MCMCのログファイル名(combined.txt)を指定する。複数のファイルを指定可能。
    
**-t TAG, --tag TAG**
    取り出す項目の tag を指定する。必須パラメータ。
    
**-d DATA_DIR, --data_dir DATA_DIR**
    データファイルをディレクトリから取得する場合(``FILE`` を指定しない場合)に、ディレクトリを指定する。
			
**--progress**
    実行時にプログレスバーを表示する。表示には tqdm ライブラリが必要。
    
**-h, --help**
    ヘルプメッセージを表示してプログラムを終了する。
