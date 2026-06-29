QUBO 問題に対するテンソル列最適化
===================================

このチュートリアルでは、TTOpt アルゴリズムを用いて 2 次制約なし 2 値最適化（QUBO）問題を解く方法を説明します。
この例は離散パラメータをもつ関数の最適化を扱います。連続パラメータをもつ関数への適用例については :doc:`ttopt` を参照してください。

前準備
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QUBO 問題のインスタンスを生成するために Python パッケージ ``qubogen`` を使用するため、本チュートリアルを実行するには事前にインストールが必要です。次のコマンドでインストールできます。

.. code-block::

    $ python3 -m pip install qubogen

また、ODAT-SE の前提パッケージとしてすでにインストールされているはずの ``numpy`` に加え、``pandas`` および ``networkx`` が必要です。

参照ソルバーとして MATLAB を使用しますが、必須ではありません。

サンプルファイルの場所
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

サンプルファイルは ``sample/qubo/ttopt`` にあります。
フォルダには以下のファイルが格納されています。

- ``run_qubo_test.py``

  メインプログラムファイル。

- ``qubo_instances.py``

  QUBO 問題インスタンスを定義するモジュール。``run_qubo_test.py`` と ``qubo_mats.py`` が共有し、両者が同一のインスタンスを解くようにします。

- ``qubo_mats.py``

  QUBO 問題インスタンスを MATLAB の ``.mat`` 形式でフォルダに出力する補助スクリプト。

- ``solve_qubo_mats_tabu.m``

  ``qubo_mats.py`` で生成した QUBO 問題インスタンスを、タブー探索を実装した MATLAB ソルバーで解く補助スクリプト。

- ``compare_results.py``

  ODAT-SE の実行結果と MATLAB ソルバーの結果を比較する補助スクリプト。

- ``qubo_mats_sol.zip``

  MATLAB ソルバーを実行するのが難しい場合のために、``solve_qubo_mats_tabu.m`` の出力結果をまとめた ZIP アーカイブ。


入力ファイルの説明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

この例では、QUBO 問題の種類ごとに入力 TOML ファイルを動的に生成しています。以下はそのテンプレートです。

.. code-block::

    [base]
    dimension = {dim}
    output_dir = "{output_dir}"

    [solver]
    name = "custom"
    function_name = "{func_name}"

    [algorithm]
    name = "ttopt"
    seed = {12345+j}

    [algorithm.param]
    max_list = {max_list}
    min_list = {min_list}

    [algorithm.ttopt]
    p_points = {p_points}
    q_points = {q_points}
    r_max = 4
    max_f_eval = 100000

``{...}`` の形式のパラメータは ``run_qubo_test.py`` 内で適切な値に置き換えられるメタ変数です。詳細は :doc:`../input` および :doc:`../algorithm/ttopt` を参照してください。``{12345+j}`` の変数 ``j`` はトライアルの通し番号を表しており、同一問題インスタンスに対して異なる乱数シードで複数回の試行を行えます。

``[base]`` セクションでは ODAT-SE 実行時のグローバルパラメータを指定します。

- ``dimension`` は最適化の次元数（状態空間上の 1 点を指定するスカラーパラメータの数）を表します。

- ``output_dir`` は ODAT-SE の実行結果を保存する場所です。

``[algorithm]`` セクションではアルゴリズムの共通パラメータを指定します。

- ``name`` はアルゴリズムの名前です（ここでは ``ttopt``）。

- ``seed`` は擬似乱数生成器の初期シードです。

``[algorithm.param]`` セクションでは探索するパラメータ空間を設定します。

- ``max_list`` は各次元の上限値のリストです。

- ``min_list`` は各次元の下限値のリストです。

``[algorithm.ttopt]`` セクションでは TTOpt アルゴリズム固有のパラメータを指定します。

- ``p_points`` は各次元のモード数であり、パラメータに割り当てたテンソル脚のボンド次元に対応します。``q_points`` と合わせて、各パラメータが取りうる値の数が決まります。1 つのパラメータが取りうる値の総数は :math:`p^q` です。

- ``q_points`` は各次元のサブモード数であり、パラメータに割り当てたテンソル脚の本数に対応します。``p_points`` と合わせて、各パラメータが取りうる値の数が決まります。1 つのパラメータが取りうる値の総数は :math:`p^q` です。

- ``r_max`` は状態空間をモデル化する暗示的 MPS 表現の最大ランクです。

- ``max_f_eval`` は最適化で使用する関数評価回数の上限です。


計算の実行
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

チュートリアルファイルが置かれているディレクトリへ移動します。

.. code-block::

   $ cd sample/qubo/ttopt

このチュートリアルでは、:math:`d=50,100,200` 次元の QUBO 問題に対して候補最小値を探します。具体的には、最大カット問題（max-cut）・最小頂点被覆問題（minimum vertex cover）・二次ナップサック問題（quadratic knapsack）を対象とします。スピン系の言葉では、これらの QUBO 型問題は :math:`d` 個のイジングスピンからなる（フラストレーションを含む可能性のある）系に対応します。使用する MPI プロセス数を引数として指定しながら、次のコマンドでテストプログラムを実行します。

.. code-block::

   $ export OMP_NUM_THREADS=1; mpiexec -np 8 python3 ./run_qubo_test.py | tee log.txt

ここでは 8 プロセスの MPI を使用しています。問題サイズが大きい場合は最適化に時間がかかることがあります。

各ベンチマーク関数について、``output/output_{func_name}_{dim}_{instance}`` 形式のフォルダが作成され、ODAT-SE の出力ファイルがすべて格納されます。各 MPI プロセスには MPI ランク番号で索引付けられたサブフォルダが割り当てられます。各サブフォルダにはそのプロセスの実行時間の詳細を含むログファイルがあります。最適化全体の履歴は出力フォルダ内の ``ttopt_history.txt`` ファイルに記録されます。

``ttopt_history.txt`` ファイルには最適化メソッドの呼び出し時に設定されたパラメータの一覧と、関数評価回数・これまでの最良点・これまでに得られた最良目的関数値の記録が含まれます。

.. code-block::

    nprocs = 8
    bounds = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    p_points = [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2]
    q_points = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1]
    r_max = -369.0
    max_f_eval = 100000
    maxvol_tol = 1.001
    maxvol_max_it = 1000
    f_eval, x_opt, f_opt
    8, [1. 1. 1. 1. 0. 0. 1. 1. 1. 1. 0. 1. 1. 0. 1. 0. 0. 0. 1. 0. 1. 0. 1. 1.
     0. 1. 0. 0. 1. 1. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 0.
     1. 0.], -277.0
    24, [1. 1. 1. 1. 0. 0. 1. 1. 1. 1. 0. 1. 1. 0. 1. 0. 0. 0. 1. 0. 1. 0. 1. 1.
     0. 1. 0. 0. 1. 1. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 0.
     1. 0.], -277.0
    ...

メイン Python スクリプトを実行すると、``qubo_results.csv`` と ``qubo_results_agg.csv`` の 2 つの CSV ファイルも生成されます。前者は特定の次元における全問題インスタンスの各トライアル結果を、後者は問題と次元ごとに得られた最小関数値を集計した結果を含みます。

得られた解の品質を評価するために、MATLAB のタブー探索ソルバーのような既存のソルバーを使った別解を比較します。``qubo_mats_sol.zip`` に含まれる結果を使う場合は次の 2 つのコマンドを省略し、アーカイブの内容を作業ディレクトリに展開してください。なお MATLAB ソルバーは確率的ソルバーであるため、実行のたびに結果が多少異なることがあります。

``run_qubo_test.py`` で使用する QUBO インスタンス行列を取得するには、次のコマンドを実行します。

.. code-block::

   $ python3 ./qubo_mats.py

次に、QUBO インスタンス行列を MATLAB ソルバーに渡します。MATLAB の対話端末またはスクリプトとして実行できます。シェルから実行する場合は次のコマンドを使用します。

.. code-block::

   $ matlab -nodisplay -nosplash -nodesktop -r "run('solve_qubo_mats_tabu.m');"

その後、補助スクリプト ``compare_results.py`` を実行して両手法の結果を比較します。

.. code-block::

   $ python3 ./compare_results.py

このスクリプトは、各インスタンスについて推定最小目的関数値の相対誤差（MATLAB ソルバーの結果を基準）を計算します。各問題タイプおよび次元ごとの相対誤差の範囲を以下の表に示します。

.. raw:: html

        <style>
        .centered {
                text-align: center;
                vertical-align: middle;
        }
        table {
                width: 60%;
                margin-left: auto;
                margin-right: auto;
                border-collapse: collapse;
        }
        thead {
                border-bottom: 2px solid black;
        }
        </style>
        <table>
                <colgroup>
                        <col>
                        <col>
                        <col style="border-right: 2px solid black;">
                        <col>
                        <col>
                </colgroup>
                <thead>
                        <tr>
                                <td class="centered">問題</td>
                                <td class="centered">次元</td>
                                <td class="centered">最大相対誤差</td>
                                <td class="centered">平均相対誤差</td>
                        </tr>
                </thead>
                <tbody>
                        <tr>
                                <td class="centered">最大カット</td>
                                <td class="centered">$$d=50$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$0$$</td>
                        </tr>
                        <tr>
                                <td class="centered">最大カット</td>
                                <td class="centered">$$d=100$$</td>
                                <td class="centered">$$4.2164\times10^{-3}$$</td>
                                <td class="centered">$$7.7219\times10^{-4}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">最大カット</td>
                                <td class="centered">$$d=200$$</td>
                                <td class="centered">$$7.5553\times10^{-2}$$</td>
                                <td class="centered">$$3.6504\times10^{-3}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">最小頂点被覆</td>
                                <td class="centered">$$d=50$$</td>
                                <td class="centered">$$3.3014\times10^{-4}$$</td>
                                <td class="centered">$$1.0065\times10^{-4}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">最小頂点被覆</td>
                                <td class="centered">$$d=100$$</td>
                                <td class="centered">$$8.1636\times10^{-5}$$</td>
                                <td class="centered">$$3.2435\times10^{-5}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">最小頂点被覆</td>
                                <td class="centered">$$d=200$$</td>
                                <td class="centered">$$3.0093\times10^{-5}$$</td>
                                <td class="centered">$$1.6078\times10^{-5}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">二次ナップサック</td>
                                <td class="centered">$$d=50$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$0$$</td>
                        </tr>
                        <tr>
                                <td class="centered">二次ナップサック</td>
                                <td class="centered">$$d=100$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$0$$</td>
                        </tr>
                        <tr>
                                <td class="centered">二次ナップサック</td>
                                <td class="centered">$$d=200$$</td>
                                <td class="centered">$$1.4448\times10^{-3}$$</td>
                                <td class="centered">$$1.8861\times10^{-4}$$</td>
                        </tr>
                </tbody>
        </table>

.. raw:: latex

        \begin{tabular}{cc|cc}
        問題 & 次元 & 最大相対誤差 & 平均相対誤差 \\
        \hline
        最大カット & $50$ & $0$ & $0$ \\
        最大カット & $100$ & $4.2164\times10^{-3}$ & $7.7219\times10^{-4}$ \\
        最大カット & $200$ & $7.5553\times10^{-2}$ & $3.6504\times10^{-3}$ \\
        最小頂点被覆 & $50$ & $3.3014\times10^{-4}$ & $1.0065\times10^{-4}$ \\
        最小頂点被覆 & $100$ & $8.1636\times10^{-5}$ & $3.2435\times10^{-5}$ \\
        最小頂点被覆 & $200$ & $3.0093\times10^{-5}$ & $1.6078\times10^{-5}$ \\
        二次ナップサック & $50$ & $0$ & $0$ \\
        二次ナップサック & $100$ & $0$ & $0$ \\
        二次ナップサック & $200$ & $1.4448\times10^{-3}$ & $1.8861\times10^{-4}$ \\
        \end{tabular}

この結果から、TTOpt は検討したすべての QUBO 問題タイプにおいて参照ソルバーと非常に近い結果を得られることがわかります。
