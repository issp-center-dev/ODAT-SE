``Algorithm``
================================

``Algorithm`` is defined as a subclass of ``odatse.algorithm.AlgorithmBase``:

.. code-block:: python

    import odatse

    class Algorithm(odatse.algorithm.AlgorithmBase):
        pass


``AlgorithmBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlgorithmBase`` class offers the following methods.

- ``__init__(self, info: odatse.Info, runner: odatse.Runner = None)``

  - Reads the common parameters from ``info`` and sets the following instance variables:

    - ``self.mpicomm: Optional[MPI.Comm]`` : ``MPI.COMM_WORLD``

    - ``self.mpisize: int`` : the number of MPI processes

    - ``self.mpirank: int`` : the rank of this process

      - When ``import mpi4py`` fails, ``self.mpicomm`` is set to ``None``, ``self.mpisize`` is set to 1, and ``self.mpirank`` is set to 0.

    - ``self.rng: np.random.Generator`` : pseudo random number generator

      - For details of the seed, see :ref:`the [algorithm] section of the input parameter <input_parameter_algorithm>`

    - ``self.dimension: int`` : the dimension of the parameter space

    - ``self.label_list: List[str]`` : the name of each axes of the parameter space

    - ``self.root_dir: pathlib.Path`` : root directory

      - It is taken from ``info.base["root_dir"]``.

    - ``self.output_dir: pathlib.Path`` : output directory

      - It is taken from ``info.base["output_dir"]``.

    - ``self.proc_dir: pathlib.Path`` : working directory of each process

      - It is set to ``self.output_dir / str(self.mpirank)``.
      - The directory will be made automatically.
      - Each process performs an optimization algorithm in this directory.

    - ``self.timer: dict[str, dict]`` : dictionary storing elapsed time

      - Three empty dictinaries, ``"prepare"``, ``"run"``, and ``"post"``, will be defined.

    - ``self.checkpoint: bool`` : enable/disable checkpointing

      - ``self.checkpoint_steps``
      - ``self.checkpoint_interval``
      - ``self.checkpoint_file``

	The parameters concerning the checkpointing feature are stored.

- ``prepare(self) -> None``

    - Prepares the algorithm.
    - It should be called before ``self.run()`` is called.

- ``run(self) -> None``

    - Performs the algorithm
    - The following steps are executed:

      #. Enter into the directory ``self.proc_dir``.
      #. Run ``self.runner.prepare()``.
      #. Run ``self._run()``.
      #. Run ``self.runner.post()``.
      #. Move to the original directory.

    - It should be called after ``self.prepare()`` is called.

- ``post(self) -> Dict``

    - Runs a post process of the algorithm, for example, writing the results into files.
    - Enters into ``self.output_dir``, calls ``self._post()``, and returns to the original directory.
    - It should be called after ``self.run()`` is called.

- ``main(self, run_mode) -> Dict``

    - Calls ``prepare``, ``run``, and ``post``.
    - Measures the elapsed times for calling functions, and writes them into a file
    - Takes an argument for the execution mode as a string. The default value is ``initialize``.

      - ``"initialize"``: start from the initial state.
      - ``"resume"``: resume from the interrupted run.
      - ``"continue"``: continue from the finished run.

      The argument contains ``"-resetrand"`` when the random number generator should be initialized.
      The behavior of "continue" depends on the algorithm.

    - Returns the result of the optimization in the form of dictionary.


``Algorithm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Algorithm`` provides a concrete description of the algorithm.
It is defined as a subclass of ``AlgorithmBase`` and should have the following methods.

- ``__init__(self, info: odatse.Info, runner: odatse.Runner = None, domain = None)``

    - The arguments ``info`` and ``runner`` should be transferred to the constructor of the base class:

        - ``super().__init__(info=info, runner=runner)``

    - Reads ``info`` and sets information.

    - If ``domain`` is given, the search region should be taken from the ``domain`` parameter.
      Otherwise, the search region should be created from ``info`` by ``odatse.domain.Region(info)`` (for continuous parameter space) or ``odatse.domain.MeshGrid(info)`` (for discrete parameter space).

- ``_prepare(self) -> None``

    - Describes pre-processes of the algorithm.

- ``_run(self) -> None``

    - Describes the algorithm body.

    - In order to obtain the value of the objective function ``f(x)`` for the search parameter ``x``, the method of Runner class should be called in the following manner:

      .. code-block:: python

	 args = (step, set)
         fx = self.runner.submit(x, args)

- ``_post(self) -> Dict``

    - Describes post-process of the algorithm.

    - Returns the result of the optimization in the form of dictionary.


Definition of ``Domain``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two classes are preprared to specify the search region.

``Region`` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Region`` is a helper class to define a continuous parameter space.

- The constructor takes an ``Info`` object, or a dictionary in ``param=`` form.

  - When the ``Info`` object is given, the lower and upper bounds of the region, the units, and the initial values are obtained from ``Info.algorithm.param`` field.

  - When the dictionary is given, the corresponding data are taken from the dictionary data.

  - For details, see :ref:`[algorithm.param] subsection for minsearch <minsearch_input_param>`

- ``Initialize(self, rnd, limitation, num_walkers)`` should be called to set the initial values.
  The arguments are the random number generator ``rng``, the constraint object ``limitation``, and the number of walkers ``num_walkers``.

``MeshGrid`` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``MeshGrid`` is a helper class to define a discrete parameter space.

- The constructor takes an ``Info`` object, or a dictionary in ``param=`` form.

  - When the ``Info`` object is given, the lower and upper bounds of the region, the units, and the initial values are obtained from ``Info.algorithm.param`` field.

  - When the dictionary is given, the corresponding data are taken from the dictionary data.

  - For details, see :ref:`[algorithm.param] subsection for mapper <mapper_input_param>`

- ``do_split(self)`` should be called to divide the grid points and distribute them to MPI ranks.

- For input and output, the following methods are provided.

  - A class method ``from_file(cls, path)`` is prepared that reads mesh data from ``path`` and creates an instance of ``MeshGrid`` class.

  - A method ``store_file(self, path)`` is prepared that writes the grid information to the file specified by ``path``.
