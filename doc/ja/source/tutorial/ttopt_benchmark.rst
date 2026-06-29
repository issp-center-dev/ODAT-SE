解析的テスト関数に対するテンソル列最適化
=========================================

このチュートリアルでは、TTOpt アルゴリズムを用いて最適化の分野でよく使われる高次元ベンチマーク関数の最小化を行う方法を説明します。
この例は連続パラメータをもつ関数の最適化を扱います。離散パラメータをもつ関数への適用例については :doc:`qubo` を参照してください。

サンプルファイルの場所
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

サンプルファイルは ``sample/analytical/ttopt_benchmark`` にあります。
フォルダには以下のファイルが格納されています。

- ``run_benchmarks.py``

  メインプログラムファイル。

入力ファイルの説明
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

この例では、各テスト関数に対して入力 TOML ファイルを動的に生成しています。以下はそのテンプレートです。

.. code-block::

    [base]
    dimension = {dim}
    output_dir = "{output_dir}"

    [solver]
    name = "analytical"
    function_name = "{func_name}"

    [algorithm]
    name = "ttopt"
    seed = 12345

    [algorithm.param]
    max_list = {max_list}
    min_list = {min_list}

    [algorithm.ttopt]
    p_points = {p_points}
    q_points = {q_points}
    r_max = 4
    max_f_eval = 100000

``{...}`` の形式のパラメータは ``run_benchmarks.py`` 内で適切な値に置き換えられるメタ変数です。詳細は :doc:`../input` および :doc:`../algorithm/ttopt` を参照してください。

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

   $ cd sample/analytical/ttopt_benchmark

このチュートリアルでは、2 次元および 10 次元の各種ベンチマーク関数の最小値を探します。使用する MPI プロセス数を引数として指定しながら、次のコマンドでベンチマーク最小化を実行します。

.. code-block::

   $ python3 ./run_benchmarks.py 8 | tee log.txt

ここでは 8 プロセスの MPI を使用しています。各最適化は数秒程度で完了します。

各ベンチマーク関数について、``output/output_{func_name}`` 形式のフォルダが作成され、ODAT-SE の出力ファイルがすべて格納されます。各 MPI プロセスには MPI ランク番号で索引付けられたサブフォルダが割り当てられます。各サブフォルダにはそのプロセスの実行時間の詳細を含むログファイルがあります。最適化全体の履歴は出力フォルダ内の ``ttopt_history.txt`` ファイルに記録されます。

``ttopt_history.txt`` ファイルには最適化メソッドの呼び出し時に設定されたパラメータの一覧と、関数評価回数・これまでの最良点・これまでに得られた最良目的関数値の記録が含まれます。

.. code-block::

    nprocs = 8
    bounds = [[-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768], [-32.768, 32.768]]
    p_points = [2 2 2 2 2 2 2 2 2 2]
    q_points = [25 25 25 25 25 25 25 25 25 25]
    r_max = 3.906300918732342e-06
    max_f_eval = 100000
    maxvol_tol = 1.001
    maxvol_max_it = 1000
    f_eval, x_opt, f_opt
    8, [ -5.58609489  10.23980011  25.61853104  22.65200946 -13.55066544
     -10.49954426  28.96521473 -19.50112851 -17.7153394   -9.98021612], 21.450699050623328
    24, [ -5.58609489  10.23980011  25.61853104  22.65200946 -13.55066544
     -10.49954426  28.96521473 -19.50112851 -17.7153394   -9.98021612], 21.450699050623328
    ...

出力をファイル（ここでは ``log.txt``）にリダイレクトすることで、最適化結果を後から確認できます。

以下の表に、各最適化関数について大域最小値と推定最小値をまとめます。

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
                                <td class="centered">関数</td>
                                <td class="centered">次元</td>
                                <td class="centered">探索範囲</td>
                                <td class="centered">最小値</td>
                                <td class="centered">推定最小値</td>
                        </tr>
                </thead>
                <tbody>
                        <tr>
                                <td class="centered">Ackley</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-32.768, 32.768]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$3.906\times10^{-6}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Alpine</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-10, 10]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$2.879\times10^{-7}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Exponential</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-1, 1]$$</td>
                                <td class="centered">$$-1$$</td>
                                <td class="centered">$$-1$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Griewank</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-600, 600]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$2.466\times10^{-6}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Himmelblau</td>
                                <td class="centered">$$d=2$$</td>
                                <td class="centered">$$[-6, 6]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$6.312\times10^{-13}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Michalewicz</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[0, \pi]$$</td>
                                <td class="centered">$$-9.66015$$</td>
                                <td class="centered">$$-9.578$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Qing</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-500, 500]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$1.500\times10^{-8}$$</td>
                        </tr>
                        <tr>
                                <td class="centered;">Rastrigin</td>
                                <td class="centered;">$$d=10$$</td>
                                <td class="centered;">$$[-5.12, 5.12]$$</td>
                                <td class="centered;">$$0$$</td>
                                <td class="centered;">$$4.620\times10^{-11}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Rosenbrock</td>
                                <td class="centered">$$d=2$$</td>
                                <td class="centered">$$[-5, 5]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$0.01175$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Schaffer</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-100, 100]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$3.606\times10^{-2}$$</td>
                        </tr>
                        <tr>
                                <td class="centered">Schwefel</td>
                                <td class="centered">$$d=10$$</td>
                                <td class="centered">$$[-500, 500]$$</td>
                                <td class="centered">$$0$$</td>
                                <td class="centered">$$1.273\times10^{-4}$$</td>
                        </tr>
                </tbody>
        </table>

.. raw:: latex

        \begin{tabular}{ccc|cc}
        関数 & 次元 & 探索範囲 & 最小値 & 推定最小値 \\
        \hline
        Ackley & $10$ & $[-32.768, 32.768]$ & $0$ & $3.906\times10^{-6}$ \\
        Alpine & $10$ & $[-10, 10]$ & $0$ & $2.879\times10^{-7}$ \\
        Exponential & $10$ & $[-1, 1]$ & $-1$ & $-1$ \\
        Griewank & $10$ & $[-600, 600]$ & $0$ & $2.466\times10^{-6}$ \\
        Himmelblau & $2$ & $[-6, 6]$ & $0$ & $6.312\times10^{-13}$ \\
        Michalewicz & $10$ & $[0, \pi]$ & $-9.66015$ & $-9.578$ \\
        Qing & $10$ & $[-500, 500]$ & $0$ & $1.500\times10^{-8}$ \\
        Rastrigin & $10$ & $[-5.12, 5.12]$ & $0$ & $4.620\times10^{-11}$ \\
        Rosenbrock & $2$ & $[-5, 5]$ & $0$ & $0.01175$ \\
        Schaffer & $10$ & $[-100, 100]$ & $0$ & $3.606\times10^{-2}$ \\
        Schwefel & $10$ & $[-500, 500]$ & $0$ & $1.273\times10^{-4}$ \\
        \end{tabular}

TTOpt は検討したすべての関数に対して大域最小値の良好な推定値を与えることがわかります。
