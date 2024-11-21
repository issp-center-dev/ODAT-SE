Introduction
================================

What is ODAT-SE ?
--------------------------------

Open Data Analysis Tool for Science and Engineering (ODAT-SE) is a framework for applying a search algorithm to a direct problem solver to find the optimal solution.
It has been developed by the name 2DMAT, and since version 3, it is organized as an open platform for data analysis by modularizing direct problem solvers and search algorithms.

As the standard direct problem solver, the experimental data analysis software for two-dimensional material structure analysis is prepared. The direct problem solver gives the deviation between the experimental data and the calculated data obtained under the given parameters such as atomic positions as a loss function used in the inverse problem. The optimal parameters are estimated by minimizing the loss function using a search algorithm. For further use, the original direct problem solver or the search algorithm can be defined by users.
In the current version, for solving a direct problem, ODAT-SE offers the wrapper of the solver for the total-reflection high-energy positron diffraction (TRHEPD) experiment[1, 2], sxrd[3], and leed[4].
As algorithms, it offers the Nelder-Mead method[5], the grid search method[6], the Bayesian optimization method[7], the replica exchange Monte Carlo method[8], and the population annealing Monte Carlo method[9-11].

In the future, we plan to add other direct problem solvers and search algorithms in ODAT-SE.


[1] As a review, see `Y. Fukaya, et al., J. Phys. D: Appl. Phys. 52, 013002 (2019) <https://iopscience.iop.org/article/10.1088/1361-6463/aadf14>`_.

[2] `T. Hanada, Y. Motoyama, K. Yoshimi, and T. Hoshi, Computer Physics Communications 277, 108371 (2022). <https://doi.org/10.1016/j.cpc.2022.108371>`_

[3] `W. Voegeli, K. Akimoto, T. Aoyama, K. Sumitani, S. Nakatani, H. Tajiri, T. Takahashi, Y. Hisada, S. Mukainakano, X. Zhang, H. Sugiyama, H. Kawata, Applied Surface Science 252, 5259 (2006). <https://doi.org/10.1016/j.apsusc.2005.12.019>`_

[4] `M.A. Van Hove, W. Moritz, H. Over, P.J. Rous, A. Wander, A. Barbieri, N. Materer, U. Starke, G.A. Somorjai, Automated determination of complex surface structures by LEED, Surface Science Reports, Volume 19, 191-229 (1993). <https://doi.org/10.1016/0167-5729(93)90011-D>`_

[5] `K. Tanaka, T. Hoshi, I. Mochizuki, T. Hanada, A. Ichimiya, and T. Hyodo, Acta. Phys. Pol. A 137, 188 (2020) <http://przyrbwn.icm.edu.pl/APP/PDF/137/app137z2p25.pdf>`_.

[6] `K. Tanaka, I. Mochizuki, T. Hanada, A. Ichimiya, T. Hyodo, and T. Hoshi, JJAP Conf. Series, 9, 011301 (2023) <https://doi.org/10.56646/jjapcp.9.0_011301>`_.

[7] `Y. Motoyama, R. Tamura, K. Yoshimi, K. Terayama, T. Ueno, and K. Tsuda,  Computer Physics Communications 278, 108405 (2022) <http://dx.doi.org/10.1016/j.cpc.2022.108405>`_

[8] `K. Hukushima and K. Nemoto, J. Phys. Soc. Japan, 65, 1604 (1996) <https://journals.jps.jp/doi/10.1143/JPSJ.65.1604>`_,  `R. Swendsen and J. Wang, Phys. Rev. Lett. 57, 2607 (1986) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.57.2607>`_.

[9] `R. M. Neal, Statistics and Computing 11, 125-139 (2001). <https://link.springer.com/article/10.1023/A:1008923215028>`_

[10] `K. Hukushima and Y. Iba, AIP Conf. Proc. 690, 200 (2003). <https://aip.scitation.org/doi/abs/10.1063/1.1632130>`_

[11] `J. Machta, Phys. Rev. E 82, 026704 (2010). <https://journals.aps.org/pre/abstract/10.1103/PhysRevE.82.026704>`_

License
--------------------------------
|  This package is distributed under `Mozilla Public License version 2.0 (MPL-2.0) <https://www.mozilla.org/en-US/MPL/2.0/>`_.

Copyright (c) <2020-> The University of Tokyo. All rights reserved.

This software was developed with the support of "Project for advancement of software usability in materials science" of The Institute for Solid State Physics, The University of Tokyo.
We hope that you cite the following reference when you publish the results using 2DMAT / ODAT-SE:

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

Version Information
--------------------------------

- ODAT-SE

  - v3.0.0: 2024-XX-XX

- 2DMAT
  
  - v2.1.0: 2022-04-08
  - v2.0.0: 2022-01-17
  - v1.0.1: 2021-04-15 
  - v1.0.0: 2021-03-12 
  - v0.1.0: 2021-02-08


Main developers
--------------------------------
ODAT-SE has been developed by following members.

- ODAT-SE v3.0.0 -

  - Y. Motoyama (The Institute for Solid State Physics, The University of Tokyo)
  - K. Yoshimi (The Institute for Solid State Physics, The University of Tokyo)
  - T. Aoyama (The Institute for Solid State Physics, The University of Tokyo)
  - T. Hoshi (National Institute for Fusion Science)

- 2DMAT v2.0.0 -

  - Y. Motoyama (The Institute for Solid State Physics, The University of Tokyo)
  - K. Yoshimi (The Institute for Solid State Physics, The University of Tokyo)
  - H. Iwamoto (Department of Applied Mathematics and Physics, Tottori University)
  - T. Hoshi (Department of Applied Mathematics and Physics, Tottori University)

- 2DMAT v0.1.0 - v1.0.1

  - Y. Motoyama (The Institute for Solid State Physics, The University of Tokyo)
  - K. Yoshimi (The Institute for Solid State Physics, The University of Tokyo)
  - T. Hoshi (Department of Applied Mathematics and Physics, Tottori University)
