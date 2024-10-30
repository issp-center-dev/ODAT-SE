``leed`` ソルバー
***********************************************

``leed`` は M.A. Van Hove氏により作成された ``SATLEED`` を用いて、原子位置などからRocking curve を計算し、実験で得られた Rocking curve からの誤差を :math:`f(x)` として返す ``Solver`` です。 ``SATLEED`` に関する詳細については [SATLEED]_ をご覧ください。

.. [SATLEED] M.A. Van Hove, W. Moritz, H. Over, P.J. Rous, A. Wander, A. Barbieri, N. Materer, U. Starke, G.A. Somorjai, Automated determination of complex surface structures by LEED, Surface Science Reports, Volume 19, 191-229 (1993). https://doi.org/10.1016/0167-5729(93)90011-D

前準備
~~~~~~~~~~~~
最初に ``SATLEED`` をインストールします。
http://www.icts.hkbu.edu.hk/VanHove_files/leed/leedsatl.zip へアクセスし、zipファイルをダウンロードします。zipファイル展開後に、所定の手続きに従いコンパイルすることで、 ``stal1.exe``, ``satl2.exe`` などの実行ファイルができます。
``SATLED`` は計算したい系の詳細によって、ソースコードのパラメータを適宜書き換える必要があります。
``sample/py2dmat/leed`` にあるサンプルを実行する場合には、 ``SATLEED`` のダウンロードから、サンプル向けのソースコードの書き換え、コンパイルまでを自動で行うスクリプト ``setup.sh`` が用意されています。::

    $ cd sample/py2dmat/leed
    $ sh ./setup.sh

``setup.sh`` を実行すると、 ``leedsatl`` ディレクトリに ``satl1.exe`` と ``satl2.exe`` が生成されます。

``py2dmat`` から ``SATLEED`` を利用するにあたっては、あらかじめ ``satl1.exe`` まで実行していることが前提となります。そのため、以下のファイルが生成されている必要があります。

- ``satl1.exe`` の入力ファイル: ``exp.d``, ``rfac.d``, ``tleed4.i``, ``tleed5.i``

- ``satl1.exe`` の出力ファイル: ``tleed.o`` , ``short.t``

``py2dmat`` はこれらをもちいて ``satl2.exe`` を実行します。

入力パラメータ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``solver`` セクション中のサブセクション
``config``,  ``reference`` を利用します。

[``config``] セクション
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``path_to_solver``

  形式: string型

  説明: ソルバー ``satl2.exe`` へのパス


[``reference``] セクション
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``path_to_base_dir``

  形式: string型

  説明: - ``exp.d``, ``rfac.d``, ``tleed4.i``, ``tleed5.i`` , ``tleed.o`` , ``short.t`` が格納されたディレクトリへのパス。

  
ソルバー用補助ファイル
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ターゲット参照ファイル
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ターゲットにするデータが格納されたファイル。 [``reference``] セクションの ``path_to_base_dir`` 中にある ``tleed4.i`` を編集します。最適化したい数値を ``optxxx`` (xxxは000, 001, 002, ...の形式で指定する三桁の整数)として指定します。なお、xxxの数字と ``py2dmat`` の最適化する値を入れる変数のリストの順番・個数は一致させる必要があります。なお、IFLAG, LSFLAGを0にしない場合はsatleed側での最適化も行われます。

以下、ファイル例を記載します。

.. code-block::

    1  0  0                          IPR ISTART LRFLAG
    1 10  0.02  0.2                  NSYM  NSYMS ASTEP VSTEP
    5  1  2  2                       NT0  NSET LSMAX LLCUT
    5                                NINSET
    1.0000 0.0000                  1      PQEX
    1.0000 2.0000                  2      PQEX
    1.0000 1.0000                  3      PQEX
    2.0000 2.0000                  4      PQEX
    2.0000 0.0000                  5      PQEX
    3                                NDIM
    opt000 0.0000 0.0000  0           DISP(1,j)  j=1,3
    0.0000 opt001 0.0000  0           DISP(2,j)  j=1,3
    0.0000 0.0000 0.0000  1           DISP(3,j)  j=1,3
    0.0000 0.0000 0.0000  0           DISP(4,j)  j=1,3
    0.0000  0                         DVOPT  LSFLAG
    3  0  0                          MFLAG NGRID NIV
    ...
   
出力ファイル
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``leed`` では、 計算時に出力されるファイルが、ランクの番号が記載されたフォルダ下に一式出力されます。
