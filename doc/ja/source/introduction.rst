はじめに
=====================

ODAT-SEとは
----------------------

Open Data Analysis Tool for Science and Engineering (ODAT-SE)は、順問題ソルバーに対して探索アルゴリズムを適用して最適解を探すためのフレームワークです。Version 2までは2DMATという名称で開発されてきましたが、Version 3からは順問題ソルバーと探索アルゴリズムをモジュール化した汎用のデータ解析プラットフォームとして整備しています。

順問題ソルバーはユーザー自身で定義することが可能です。
標準的な順問題ソルバーとしては2次元物質構造解析向け実験データ解析ソフトウェアが用意されています。
順問題ソルバーでは原子位置などをパラメータとし得られたデータと実験データとのずれを損失関数として与えます。
探索アルゴリズムを組み合わせ、この損失関数を最小化することで、最適なパラメータを推定します。
現バージョンでは、順問題ソルバーとして量子ビーム回折実験の全反射高速陽電子回折法(Total-reflection high-energy positron diffraction: TRHEPD)[1, 2], 表面X線回折法(sxrd)[3], 低速電子線回折法(leed)[4]に対応しており、探索アルゴリズムはNelder-Mead法[5], グリッド型探索法[6], ベイズ最適化[7], レプリカ交換モンテカルロ法[8], ポピュレーションアニーリングモンテカルロ法[9, 10, 11]が実装されています。

今後は、本フレームワークをもとにより多くの順問題ソルバーおよび探索アルゴリズムを実装していく予定です。

[1] レビューとして， `Y．Fukaya, et al., J. Phys. D: Appl. Phys. 52, 013002 (2019) <https://iopscience.iop.org/article/10.1088/1361-6463/aadf14>`_;
`兵頭俊夫，「全反射高速陽電子回折　(TRHEPD)による表面構造解析」，固体物理 53, 705 (2018) <https://www.agne.co.jp/kotaibutsuri/kota1053.htm>`_.

[2] `T. Hanada, Y. Motoyama, K. Yoshimi, and T. Hoshi, Computer Physics Communications 277, 108371 (2022). <https://doi.org/10.1016/j.cpc.2022.108371>`_

[3] `W. Voegeli, K. Akimoto, T. Aoyama, K. Sumitani, S. Nakatani, H. Tajiri, T. Takahashi, Y. Hisada, S. Mukainakano, X. Zhang, H. Sugiyama, H. Kawata, Applied Surface Science 252, 5259 (2006). <https://doi.org/10.1016/j.apsusc.2005.12.019>`_

[4] `M.A. Van Hove, W. Moritz, H. Over, P.J. Rous, A. Wander, A. Barbieri, N. Materer, U. Starke, G.A. Somorjai, Automated determination of complex surface structures by LEED, Surface Science Reports, Volume 19, 191-229 (1993). <https://doi.org/10.1016/0167-5729(93)90011-D>`_

[5] `K. Tanaka, T. Hoshi, I. Mochizuki, T. Hanada, A. Ichimiya, and T. Hyodo, Acta. Phys. Pol. A 137, 188 (2020). <http://przyrbwn.icm.edu.pl/APP/PDF/137/app137z2p25.pdf>`_.

[6] `K. Tanaka, I. Mochizuki, T. Hanada, A. Ichimiya, T. Hyodo, and T. Hoshi, JJAP Conf. Series, 9, 011301 (2023). <https://doi.org/10.56646/jjapcp.9.0_011301>`_.

[7] `Y. Motoyama, R. Tamura, K. Yoshimi, K. Terayama, T. Ueno, and K. Tsuda,  Computer Physics Communications 278, 108405 (2022). <http://dx.doi.org/10.1016/j.cpc.2022.108405>`_

[8] `K. Hukushima and K. Nemoto, J. Phys. Soc. Japan, 65, 1604 (1996). <https://journals.jps.jp/doi/10.1143/JPSJ.65.1604>`_,  `R. Swendsen and J. Wang, Phys. Rev. Lett. 57, 2607 (1986) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.57.2607>`_.

[9] `R. M. Neal, Statistics and Computing 11, 125-139 (2001). <https://link.springer.com/article/10.1023/A:1008923215028>`_

[10] `K. Hukushima and Y. Iba, AIP Conf. Proc. 690, 200 (2003). <https://aip.scitation.org/doi/abs/10.1063/1.1632130>`_

[11] `J. Machta, Phys. Rev. E 82, 026704 (2010). <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.82.026704>`_

ライセンス
----------------------
| 本ソフトウェアのプログラムパッケージおよびソースコード一式は
  `Mozilla Public License version 2.0 (MPL-2.0) <https://www.mozilla.org/en-US/MPL/2.0/>`_ に準じて配布されています。

Copyright (c) <2020-> The University of Tokyo. All rights reserved.

本ソフトウェアは2020年度・2021年度・2024年度 東京大学物性研究所 ソフトウェア高度化プロジェクトの支援を受け開発されました。
2DMAT / ODAT-SEを引用する際には以下の文献を引用してください。

`"Data-analysis software framework 2DMAT and its application to experimental measurements for two-dimensional material structures", Y. Motoyama, K. Yoshimi, I. Mochizuki, H. Iwamoto, H. Ichinose, and T. Hoshi, Computer Physics Communications 280, 108465 (2022) <https://doi.org/10.1016/j.cpc.2022.108465>`_.

BibTeX::

   @article{MOTOYAMA2022108465,
     title = {Data-analysis software framework 2DMAT and its application to experimental measurements for two-dimensional material structures},
     journal = {Computer Physics Communications},
     volume = {280},
     pages = {108465},
     year = {2022},
     issn = {0010-4655},
     doi = {https://doi.org/10.1016/j.cpc.2022.108465},
     url = {https://www.sciencedirect.com/science/article/pii/S0010465522001849},
     author = {Yuichi Motoyama and Kazuyoshi Yoshimi and Izumi Mochizuki and Harumichi Iwamoto and Hayato Ichinose and Takeo Hoshi}
   }



バージョン履歴
----------------------

- ODAT-SE

  - v3.0.0 : 2024-11-25

- 2DMAT

  - v2.1.0 : 2022-04-08
  - v2.0.0 : 2022-01-17
  - v1.0.1 : 2021-04-15
  - v1.0.0 : 2021-03-12
  - v0.1.0 : 2021-02-08

主な開発者
----------------------
ODAT-SEは以下のメンバーで開発しています.

- ODAT-SE v3.0.0 -

  - 本山 裕一 (東京大学 物性研究所)
  - 吉見 一慶 (東京大学 物性研究所)
  - 青山 龍美 (東京大学 物性研究所)
  - 星　 健夫 (核融合科学研究所)

- 2DMAT v2.0.0 -

  - 本山 裕一 (東京大学 物性研究所)
  - 吉見 一慶 (東京大学 物性研究所)
  - 岩本 晴道 (鳥取大学 大学院工学研究科)
  - 星　 健夫 (鳥取大学 大学院工学研究科)

- 2DMAT v0.1.0 -

  - 本山 裕一 (東京大学 物性研究所)
  - 吉見 一慶 (東京大学 物性研究所)
  - 星　 健夫 (鳥取大学 大学院工学研究科)
