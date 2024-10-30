共通事項
==========

``Solver`` と ``Algorithm`` に共通する事柄について説明します。

``py2dmat.Info``
~~~~~~~~~~~~~~~~~~
入力パラメータを扱うためのクラスです。
インスタンス変数として次の4つの ``dict`` を持ちます。

- ``base``

    - ディレクトリ情報など、プログラム全体で共通するパラメータ

- ``solver``

    - ``Solver`` が用いる入力パラメータ

- ``algorithm``

    - ``Algorithm`` が用いる入力パラメータ

- ``runner``

    - ``Runner`` が用いる入力パラメータ


``Info`` は ``base``, ``solver``, ``algorithm``, ``runner`` の4つのキーを持つような ``dict`` を渡して初期化出来ます。

- ``base`` について

    - 要素として計算のルートディレクトリ ``root_dir`` と出力のルートディレクトリ ``output_dir`` が自動で設定されます

    - ルートディレクトリ ``root_dir``

        - 絶対パスに変換されたものがあらためて ``root_dir`` に設定されます
        - 先頭の ``~`` はホームディレクトリに展開されます
        - デフォルトはカレントディレクトリ ``"."`` です
        - 具体的には次のコードが実行されます ::

            p = pathlib.Path(base.get("root_dir", "."))
            base["root_dir"] = p.expanduser().absolute()

    - 出力ディレクトリ ``output_dir``

        - 先頭の ``~`` はホームディレクトリに展開されます
        - 絶対パスが設定されていた場合はそのまま設定されます
        - 相対パスが設定されていた場合、 ``root_dir`` を起点とした相対パスとして解釈されます
        - デフォルトは ``"."`` 、つまり ``root_dir`` と同一ディレクトリです
        - 具体的には次のコードが実行されます ::

            p = pathlib.Path(base.get("work_dir", "."))
            p = p.expanduser()
            base["work_dir"] = base["root_dir"] / p


``py2dmat.Message``
~~~~~~~~~~~~~~~~~~~~~~
``Algorithm`` から ``Runner`` を介して ``Solver`` に渡されるクラスです。
次の3つのインスタンス変数を持ちます。

- ``x: np.ndarray``

    - 探索中のパラメータ座標

- ``step: int``

    - 何個目のパラメータであるか
    - 例えば ``exchange`` ではステップ数で、 ``mapper`` ではパラメータの通し番号。

- ``set: int``

    - 何巡目のパラメータであるか
    - 例えば ``min_search`` では最適化（1巡目）後にステップごとの最適値を再計算します（2巡目）。

``py2dmat.Runner``
~~~~~~~~~~~~~~~~~~~~~~~~~~
``Algorithm`` と ``Solver`` とをつなげるためのクラスです。
コンストラクタ引数として ``Solver`` のインスタンスと ``Info`` のインスタンス、パラメータの変換ルーチン ``mapping : Callable[[np.ndarray], np.ndarray]`` を取ります。

``submit(self, message: py2dmat.Message) -> float`` メソッドでソルバーを実行し、結果を返します。
探索パラメータを ``x`` として、 目的関数 ``fx = f(x)`` を得たい場合は以下のようにします ::

    message = py2dmat.Message(x, step, set)
    fx = runner.submit(message)

``submit`` メソッドは ``mapping`` を用いて、探索アルゴリズムのパラメータ ``x`` から実際にソルバーが使う入力 ``y = mapping(x)`` を得ます。
``mapping`` を省略した場合 (``None`` を渡した場合)、変換ルーチンとしてアフィン写像 :math:`y=Ax+b` が用いられます (``py2dmat.util.mapping.Affine(A,b)``)。
``A, b`` の要素は ``info`` で与えられます。
その他、 ``Runner`` で使われる ``info`` の詳細は :doc:`../input` を参照してください。
