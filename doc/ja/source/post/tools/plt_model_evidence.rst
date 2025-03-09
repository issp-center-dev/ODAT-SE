plt_model_evidence.py
=====================

NAME
----
model evidence を計算する

SYNOPSIS
--------

.. code-block:: bash

   python3 plt_model_evidence.py [OPTION]... -n NDATA FILEs


DESCRIPTION
-----------

PAMC の出力ファイル FILE から beta および分配関数の値を取り出し、model evidence を計算する。
結果は標準出力に書き出される。また、結果をプロットした図をファイルに出力する。

複数の FILE を指定した場合は、それらの model evidence の平均と分散を出力し、エラーバー付きのプロットを生成する。


**FILE**
    PAMCの出力ファイル名 (fx.txt)。
    
**-n NDATA, --ndata NDATA**
    各スポットのデータ点の数をカンマ区切りの整数値で指定する。必須パラメータ。
    
**-w WEIGHT, --weight WEIGHT**
    スポットの相対重みをカンマ区切りの数値で指定する。重みは和が 1.0 になるように自動で規格化される。重みの数値の個数とデータ点の個数は一致させる必要がある。
    
**-V VOLUME, --Volume VOLUME**
    事前確率分布の normalization (定義域の体積 :math:`V_\Omega`) を指定する。デフォルトは 1.0。
    
**-f RESULT, --result RESULT**
    model evidence の値を出力するファイル名。デフォルトは model_evidence.txt 。
    
**-o OUTPUT, --output OUTPUT**
    model evidence のプロットを出力するファイル名。出力形式は拡張子を元に設定され、matplotlib がサポートする形式を指定可能。デフォルトは model_evidence.png 。
    
**-h, --help**
    ヘルプメッセージを表示してプログラムを終了する。

