``Algorithm``
================================

``Algorithm`` is defined as a subclass of ``odatse.algorithm.AlgorithmBase``:

.. code-block:: python

    import odatse

    class Algorithm(odatse.algorithm.AlgorithmBase):
        pass


``AlgorithmBase``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AlgorithmBase`` provides the common infrastructure for all algorithms.

Lifecycle
^^^^^^^^^

``main()`` drives the three-phase lifecycle by calling internal framework wrappers:

.. code-block:: none

    main()
      ├── _prepare()   runner.prepare() → dispatch(init/resume/continue) → prepare()
      ├── _run()       run()
      └── _post()      post() → runner.post()

Subclasses implement the *extension points* – the methods **without** a leading
underscore: ``_initialize()``, ``prepare()`` (optional), ``run()``, and ``post()``.
The underscore-prefixed wrappers ``_prepare``, ``_run``, ``_post`` are framework
internals and must **not** be overridden in subclasses.

Instance variables set by ``__init__``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``__init__(self, info: odatse.Info, runner: odatse.Runner = None, mpicomm: Optional["MPI.Comm"] = None)``

  Reads the common parameters from ``info`` and sets the following instance variables:

  - ``self.mpicomm: Optional[MPI.Comm]`` : MPI communicator (``mpi4py.MPI.Comm``).

    - If ``mpicomm`` is given, it is used directly.
    - Otherwise ``mpi4py.MPI.COMM_WORLD`` is used when ``mpi4py`` is available; ``None`` for serial execution.

  - ``self.mpisize: int`` : number of MPI processes (``1`` when MPI is unavailable).

  - ``self.mpirank: int`` : rank of this process (``0`` when MPI is unavailable).

  - ``self.rng: np.random.RandomState`` : pseudo-random number generator.

    See :ref:`the [algorithm] section of the input parameter <input_parameter_algorithm>` for seed details.

  - ``self.dimension: int`` : dimension of the parameter space.

  - ``self.label_list: list[str]`` : name of each parameter axis.

  - ``self.root_dir: pathlib.Path`` : root directory (from ``info.base["root_dir"]``).

  - ``self.output_dir: pathlib.Path`` : output directory (from ``info.base["output_dir"]``).

  - ``self.proc_dir: pathlib.Path`` : per-process working directory.

    - Set to ``self.output_dir / str(self.mpirank)``.
    - Created automatically.
    - ``run()`` is called from this directory.

  - ``self.timer: dict[str, dict]`` : elapsed-time dictionary.

    Sub-dictionaries ``"prepare"``, ``"run"``, and ``"post"`` are pre-created.

  - ``self.checkpoint: bool`` : whether checkpointing is enabled.
  - ``self.checkpoint_file: str`` : absolute path to the checkpoint file (default: ``<proc_dir>/status.pickle``).
  - ``self.checkpoint_steps: int`` : save a checkpoint every this many steps.
  - ``self.checkpoint_interval: float`` : save a checkpoint every this many seconds.

  - ``self.mode: str`` : run mode string (``"initial"``, ``"resume"``, ``"continue"``, or with ``"-resetrand"`` suffix).

Framework wrappers (do not override)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``_prepare(self) -> None``

  Calls ``runner.prepare()``, dispatches init/resume/continue, then calls ``prepare()``.
  The checkpoint dispatch is handled automatically:

  - ``mode`` starts with ``"init"`` → ``_initialize()`` is called.
  - ``mode`` starts with ``"resume"`` or ``"continue"`` → ``_load_state()`` is called.

  Do **not** override this method. Implement ``prepare()`` instead.

- ``_run(self) -> None``

  Enters ``proc_dir`` and calls ``run()``.
  Runner calls are handled by ``_prepare()`` and ``_post()``; this wrapper performs no runner invocations.

  Do **not** override this method. Implement ``run()`` instead.

- ``_post(self) -> dict``

  Enters ``output_dir``, calls ``post()``, then calls ``runner.post()``.

  Do **not** override this method. Implement ``post()`` instead.

- ``main(self) -> dict``

  Calls ``_prepare()``, ``_run()``, and ``_post()`` in sequence with timing and MPI barriers.
  Returns the result of the optimization as a dictionary.

Checkpoint helpers
^^^^^^^^^^^^^^^^^^^

- ``_save_state(self, filename) -> None``

  Saves a checkpoint snapshot to ``filename`` using ``__getstate__()`` and versioned pickle storage.
  Override **only** when extra files must be written (e.g. an external model), calling ``super()._save_state(filename)`` first.

- ``_load_state(self, filename, mode="resume", restore_rng=True) -> None``

  Loads a checkpoint snapshot and calls ``_apply_state()``.
  Override **only** when extra files must be read, calling ``super()._load_state(...)`` first.

- ``_apply_state(self, data, mode="resume", restore_rng=True) -> None``

  Restores the base algorithm state (MPI validation, timer, parameters).
  Override in subclasses to restore algorithm-specific fields and to implement ``"continue"``-mode semantics.
  Always call ``super()._apply_state(data, mode=mode, restore_rng=restore_rng)`` first.

- ``__getstate__(self) -> dict``

  Returns a checkpoint snapshot by collecting all fields listed in ``_checkpoint_attrs`` across the MRO.
  Override **only** when extra non-attribute data must be saved (e.g. a global RNG or an external policy object).


``Algorithm`` (subclass)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Algorithm`` provides the concrete description of the algorithm.
It is defined as a subclass of ``AlgorithmBase`` and must implement the following.

``__init__``
^^^^^^^^^^^^^

.. code-block:: python

    def __init__(self, info: odatse.Info, runner: odatse.Runner = None,
                 run_mode: str = "initial", mpicomm=None):
        super().__init__(info=info, runner=runner, run_mode=run_mode, mpicomm=mpicomm)
        # read algorithm-specific parameters from info ...

Pass ``info``, ``runner``, ``run_mode``, and ``mpicomm`` to the base class constructor.
Read algorithm-specific parameters from ``info`` **after** calling ``super().__init__()``,
because the base constructor sets the attributes (``mpisize``, ``rng``, ``proc_dir``, …)
that the subclass may need.

``_initialize`` (required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def _initialize(self) -> None:
        # Set up the algorithm state for a fresh run.
        # Do NOT call the runner here – evaluation happens in run().
        self.istep = 0
        self.best_fx = np.inf
        ...

Called by the framework when ``mode`` starts with ``"init"``.
Must not use the runner (the initial evaluation should be done in ``run()``).

``prepare`` (optional)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def prepare(self) -> None:
        # Called after the checkpoint dispatch and before the main loop.
        # Good place to initialise timers or open output files.
        self.timer["run"]["submit"] = 0.0

Called after ``_initialize()`` or ``_load_state()`` and before ``run()``.
The default implementation does nothing; override only when needed.

``run`` (required)
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def run(self) -> None:
        # The checkpoint dispatch (init/resume/continue) has already been
        # performed by _prepare(); start the main loop directly.

        # For "init" mode, perform the initial evaluation here.
        if self.mode.startswith("init"):
            self.fx = self._evaluate(self.state)
            ...

        # Main loop
        while self.istep < self.numsteps:
            ...
            # Evaluate the objective function:
            args = (self.istep, 0)
            fx = self.runner.submit(x, args)
            ...
            self.istep += 1

            # Save a checkpoint periodically:
            if self.checkpoint:
                time_now = time.time()
                if self.istep >= next_checkpoint_step or time_now >= next_checkpoint_time:
                    self._save_state(self.checkpoint_file)
                    next_checkpoint_step = self.istep + self.checkpoint_steps
                    next_checkpoint_time = time_now + self.checkpoint_interval

        if self.checkpoint:
            self._save_state(self.checkpoint_file)

The algorithm body.
The checkpoint dispatch is already done; ``run()`` can check ``self.mode`` only for
actions that are specific to the very first evaluation step (``mode.startswith("init")``).

To evaluate the objective function for parameter ``x``:

.. code-block:: python

    args = (step, set)
    fx = self.runner.submit(x, args)

``post`` (required)
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def post(self) -> dict:
        # Write results to files, gather from MPI ranks, …
        return {"x": self.xopt, "fx": self.best_fx}

Post-processes the algorithm results and returns them as a dictionary.
Called from ``output_dir``.

Checkpoint fields: ``_checkpoint_attrs``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Declare a class variable ``_checkpoint_attrs`` listing the attribute names that must
be saved and restored at each checkpoint:

.. code-block:: python

    class Algorithm(odatse.algorithm.AlgorithmBase):
        _checkpoint_attrs: list[str] = ["istep", "best_x", "best_fx"]

``__getstate__()`` in the base class walks the MRO and collects every field listed in
``_checkpoint_attrs`` automatically; no override is needed for simple cases.

For custom restore logic (e.g. ``"continue"``-mode semantics), override ``_apply_state()``:

.. code-block:: python

    def _apply_state(self, data: dict, mode: str = "resume",
                     restore_rng: bool = True) -> None:
        super()._apply_state(data, mode=mode, restore_rng=restore_rng)
        # restore algorithm-specific fields:
        self.istep  = data["istep"]
        self.best_x = data["best_x"]
        self.best_fx = data["best_fx"]
        if mode == "continue":
            # extend the schedule or advance counters as needed
            ...


Minimal working example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import time
    import odatse

    class Algorithm(odatse.algorithm.AlgorithmBase):
        """Grid-search algorithm example."""

        _checkpoint_attrs: list[str] = ["icount", "best_x", "best_fx", "results"]

        def __init__(self, info, runner=None, run_mode="initial", mpicomm=None):
            super().__init__(info=info, runner=runner,
                             run_mode=run_mode, mpicomm=mpicomm)
            self.mesh = [...]   # read from info

        def _initialize(self) -> None:
            self.icount = 0
            self.best_fx = np.inf
            self.best_x = None
            self.results = []

        def prepare(self) -> None:
            self.timer["run"]["submit"] = 0.0

        def run(self) -> None:
            next_chk_step = self.icount + self.checkpoint_steps
            next_chk_time = time.time() + self.checkpoint_interval

            while self.icount < len(self.mesh):
                x = np.array(self.mesh[self.icount])
                args = (self.icount, 0)
                time_sta = time.perf_counter()
                fx = self.runner.submit(x, args)
                self.timer["run"]["submit"] += time.perf_counter() - time_sta

                self.results.append((x, fx))
                if fx < self.best_fx:
                    self.best_fx, self.best_x = fx, x.copy()
                self.icount += 1

                if self.checkpoint:
                    now = time.time()
                    if self.icount >= next_chk_step or now >= next_chk_time:
                        self._save_state(self.checkpoint_file)
                        next_chk_step = self.icount + self.checkpoint_steps
                        next_chk_time = now + self.checkpoint_interval

            if self.checkpoint:
                self._save_state(self.checkpoint_file)

        def post(self) -> dict:
            if self.mpirank == 0:
                with open("result.txt", "w") as f:
                    f.write(f"fx = {self.best_fx}\n")
            return {"x": self.best_x, "fx": self.best_fx}


Definition of ``Domain``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two classes are provided to specify the search region.

``Region`` class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Region`` is a helper class to define a continuous parameter space.

- The constructor takes an ``Info`` object, or a dictionary in ``param=`` form.

  - When the ``Info`` object is given, the lower and upper bounds of the region, the units, and the initial values are obtained from ``Info.algorithm.param`` field.

  - When the dictionary is given, the corresponding data are taken from the dictionary data.

  - For details, see :ref:`[algorithm.param] subsection for minsearch <minsearch_input_param>`

- ``initialize(self, rng, limitation, num_walkers)`` should be called to set the initial values.
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
