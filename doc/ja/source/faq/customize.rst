========================================
カスタマイズ
========================================

自分の関数を最適化したい
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`/customize/tutorial_solver` に、独自の目的関数を定義して ODAT-SE で最小化するチュートリアルがあります。
コピペして動かせる完全な例を提供しています。


外部プログラムをソルバーとして使いたい
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``evaluate`` メソッドの中で ``subprocess`` を使って外部プログラムを呼び出すことができます。

.. code-block:: python

    import subprocess

    class MySolver(odatse.solver.SolverBase):
        def evaluate(self, x, args=()):
            # パラメータをファイルに書き出す
            with open("params.dat", "w") as f:
                for xi in x:
                    f.write(f"{xi}\n")

            # 外部プログラムを実行
            subprocess.run(["./my_program", "params.dat"], check=True)

            # 結果を読み取る
            with open("result.dat") as f:
                fx = float(f.read().strip())

            return fx

``evaluate`` メソッドは、プロセスごとの作業ディレクトリ ``proc_dir`` をカレントディレクトリとして呼び出されます。
そのため、MPI 並列時にもファイルの競合が起こりにくくなっています。

より本格的なパッケージとして開発する場合は、 `ODAT-SE Gallery <https://isspns-gitlab.issp.u-tokyo.ac.jp/takeohoshi/odat-se-gallery>`_ に含まれるソルバーテンプレートの利用もご検討ください（ :doc:`/gallery` 参照）。


実際の解析例を参考にしたい
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ODAT-SE Gallery <https://isspns-gitlab.issp.u-tokyo.ac.jp/takeohoshi/odat-se-gallery>`_ に、TRHEPD, SXRD, LEED, XAFS の各手法を用いた解析例が公開されています。
入力ファイル、実行スクリプト、可視化スクリプトが含まれており、そのまま実行して結果を確認できます。
詳細は :doc:`/gallery` を参照してください。
