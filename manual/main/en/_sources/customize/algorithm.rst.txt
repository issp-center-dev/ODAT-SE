``Algorithm``
======================

``Algorithm`` is defined as a subclass of ``py2dmat.algorithm.AlgorithmBase``  ::

    import py2dmat

    class Algorithm(py2dmat.algorithm.AlgorithmBase):
        ...

``AlgorithmBase``
~~~~~~~~~~~~~~~~~~~

``AlgorithmBase`` class offers the following methods ::

- ``__init__(self, info: py2dmat.Info, runner: py2dmat.Runner = None)``

    - Reads the common parameters from ``info`` and sets the following instance variables:

        - ``self.mpicomm: Optional[MPI.Comm]`` : ``MPI.COMM_WORLD``

            - When ``import mpi4py`` fails, this will be ``None``.

        - ``self.mpisize: int`` : the number of MPI processes

            - When ``import mpi4py`` fails, this will be ``1``.

        - ``self.mpirank: int`` : the rank of this process

            - When ``import mpi4py`` fails, this will be ``0``.

        - ``self.rng: np.random.Generator`` : pseudo random number generator

            - For details of the seed, please see :ref:`the [algorithm] section of the input parameter <input_parameter_algorithm>`

        - ``self.dimension: int`` : the dimension of the parameter space
        - ``self.label_list: List[str]`` : the name of each axes of the parameter space
        - ``self.root_dir: pathlib.Path`` : root directory

            - ``info.base["root_dir"]``

        - ``self.output_dir: pathlib.Path`` : output directory

            - ``info.base["root_dir"]``

        - ``self.proc_dir: pathlib.Path`` : working directory of each process

            - ``self.output_dir / str(self.mpirank)``
            - Directory will be made automatically
            - Each process performs an optimization algorithm in this directory

        - ``self.timer: dict[str, dict]`` : dictionary storing elapsed time

            - Three empty dictinaries, ``"prepare"``, ``"run"``, and ``"post"`` will be defined

- ``prepare(self) -> None``

    - Prepares the algorithm
    - It should be called before ``self.run()`` is called
    - It calls ``self._prepare()``

- ``run(self) -> None``

    - Performs the algorithm
    - Enters into ``self.proc_dir``, calls ``self._run()``, and returns to the original directory.

- ``post(self) -> None``

    - Runs a post process of the algorithm, for example, write the result into files
    - It should be called after ``self.run()`` is called
    - Enters into ``self.output_dir``, calls ``self._post()``, and returns to the original directory.

- ``main(self) -> None``

    - Calls ``prepare``, ``run``, and ``post``
    - Measures the elapsed times for calling each function, and write them into file

- ``_read_param(self, info: py2dmat.Info) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]``

    - Helper method for initializing defining the continuous parameter space
    - Reads ``info.algorithm["param"]`` and returns the followings:

        - Initial value
        - Lower bound
        - Upper bound
        - Unit

    - For details, see :ref:`[algorithm.param] subsection for minsearch <minsearch_input_param>`

- ``_meshgrid(self, info: py2dmat.Info, split: bool = False) -> Tuple[np.ndarray, np.ndarray]``

    - Helper method for initializing defining the discrete parameter space
    - Reads ``info.algorithm["param"]`` and returns the followings:

        - ``N`` points in the ``D`` dimensinal space as a ``NxD`` matrix
        - IDs of points as a ``N`` dimensional vector

    - If ``split`` is ``True``, the set of points is scatterred to MPI processes

    - For details, see :ref:`[algorithm.param] subsection for mapper <mapper_input_param>`

``Algorithm``
~~~~~~~~~~~~~~~~

In ``Algorithm``, the following methods should be defined:

- ``__init__(self, info: py2dmat.Info, runner: py2dmat.Runner = None)``

    - Please transfer the arguments to the constructor of the base class:

        - ``super().__init__(info=info, runner=runner)``

    - Reads ``info`` and sets information

- ``_prepare(self) -> None``

    - Pre process

- ``_run(self) -> None``

    - The algorithm itself
    - In this method, you can calculate ``f(x)`` from a parameter ``x`` as the following::

        message = py2dmat.Message(x, step, set)
        fx = self.runner.submit(message)

- ``_post(self) -> None``

    - Post process
