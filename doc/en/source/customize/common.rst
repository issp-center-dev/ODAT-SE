Common items
================================

In this section, the components used throughout the program are described.


``odatse.Info``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This class treats the input parameters.
It contains the following four instance variables.

- ``base`` : ``dict[str, Any]``

  - Parameters for the whole program, such as the directory where the output will be written.

- ``solver`` : ``dict[str, Any]``

  - Parameters for ``Solver``

- ``algorithm`` : ``dict[str, Any]``

  - Parameters for ``Algorithm``

- ``runner`` : ``dict[str, Any]``

  - Parameters for ``Runner``


An instance of ``Info`` is initialized by passing a ``dict`` which has the following four sub dictionaries, ``base``, ``solver``, ``algorithm``, and ``runner``. (Some of them can be omitted.)
Each sub dictionary is set to the corresponding field of ``Info``.
Alternatively, it can be created by passing to the class method ``from_file`` a path to an input file in TOML format.


``base`` items
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As items of ``base`` field, ``root_dir`` indicating the root directory of the calculation, and ``output_dir`` for output results will be set automatically as follows.

- Root directory ``root_dir``

  - The default value is ``"."`` (the current directory).
  - The value of ``root_dir`` will be converted to an absolute path.
  - The leading ``~`` will be expanded to the user's home directory.
  - Specifically, the following code is executed:

    .. code-block:: python

       p = pathlib.Path(base.get("root_dir", "."))
       base["root_dir"] = p.expanduser().absolute()

- Output directory ``output_dir``

  - The leading ``~`` will be expanded to the user's home directory.
  - If an absolute path is given, it is set as-is.
  - If a relative path is given, it is regarded to be relative to ``root_dir``.
  - The default value is ``"."``, that is, the same as ``root_dir``
  - Specifically, the following code is executed:

    .. code-block:: python

       p = pathlib.Path(base.get("output_dir", "."))
       p = p.expanduser()
       base["output_dir"] = base["root_dir"] / p


``odatse.Runner``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Runner`` is a class that connects ``Algorithm`` and ``Solver``.
The constructor of ``Runner`` takes instances of ``Solver``, ``Info``, ``Mapping``, and ``Limitation``.
If the instance of ``Mapping`` is omitted, ``TrivialMapping``, which performs no transformation, is assumed.
If the instance of ``Limitation`` is omitted, ``Unlimited`` is assumed, which imposes no constraints.

``submit(self, x: np.ndarray, args: Tuple[int,int]) -> float`` method invokes the solver and returns the value of objective function ``f(x)``.
``submit`` internally uses the instance of ``Limitation`` to check whether the search parameter ``x`` satisfies the constraints. Then, it applies the instance of ``Mapping`` to obtain from ``x`` the input ``y = mapping(x)`` that is actually used by the solver.

See :doc:`../input/index` for details of how and which components of ``info`` the ``Runner`` uses.


``odatse.Mapping``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Mapping`` is a class that describes mappings from the search parameters of the inverse problem analysis algorithms to the variables of the direct problem solvers.
It is defined as a function object class that has ``__call__(self, x: np.ndarray) -> np.ndarray`` method.
In the current version, a trivial transformation ``TrivialMapping`` and an affine mapping ``Affine`` are defined.

``TrivialMapping``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TrivialMapping`` provides a trivial transformation :math:`x\to x`, that is, no transformation.
It is used as the default argument of the ``Runner`` class.

``Affine``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Affine`` provides an affine mapping :math:`x \to y = A x + b`.
The coefficients ``A`` and ``b`` should be given as constructor arguments, or passed as dictionary elements through the ``from_dict`` class method.
When they are specified in the ODAT-SE input file, see the input file section of the manual for the format of the parameters.


``odatse.Limitation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Limitation`` is a class that describes constraints on the :math:`N` dimensional parameter space :math:`x` searched by the inverse problem analysis algorithms.
It is defined as a class that has the method ``judge(self, x: np.ndarray) -> bool``.
In the current version, the ``Unlimited`` class, which imposes no constraint, and the ``Inequality`` class, which represents linear inequality constraints, are provided.

``Unlimited``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Unlimited`` represents that no constraint is imposed.
``judge`` method always returns ``True``.
It is used as the default argument of the ``Runner`` class.


``Inequality``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Inequality`` is a class that expresses :math:`M` constraints imposed on :math:`N` dimensional search parameters :math:`x` in the form :math:`A x + b > 0` where :math:`A` is an :math:`M \times N` matrix and :math:`b` is an :math:`M`-dimensional vector.

The coefficients ``A`` and ``b`` should be given as constructor arguments, or passed as dictionary elements through the ``from_dict`` class method.
When they are specified in the ODAT-SE input file, see the input file section of the manual for the format of the parameters.


``odatse.initialize``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``initialize(argv=None) -> (Info, str)`` is an initialization function that parses command-line style arguments and loads the input file in one step.
It interprets the same arguments as the ``odatse`` command (the path to the input file and ``--init`` / ``--resume`` / ``--cont`` / ``--reset_rand`` / ``--nalg`` / ``--nsolve``; see :doc:`../manual/command` for details), and returns a pair of an ``Info`` instance and a run-mode string ``run_mode``. It also calls ``odatse.mpi.setup()`` internally, unless ``setup()`` has already been called (``odatse.mpi.ready()`` is ``True``): then the existing partition is kept, and ``--nalg`` / ``--nsolve``, if given, must agree with it.

- When ``argv`` is omitted (``None``), ``sys.argv[1:]`` is interpreted.
  When embedding odatse in a script that has its own argument handling, pass an explicit list as ``argv`` to initialize without depending on ``sys.argv``.

  .. code-block:: python

      info, run_mode = odatse.initialize(["input.toml", "--resume"])

- ``run_mode`` is one of ``"initial"``, ``"resume"``, and ``"continue"`` (with the suffix ``"-resetrand"`` appended when ``--reset_rand`` is specified).
  Passing it to the ``run_mode`` argument of the ``Algorithm`` constructor enables the restart features also in your own scripts.


``odatse.mpi``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A module that provides access to the MPI communicators.
It works as a non-MPI stub when mpi4py is not installed or when the environment variable ``ODATSE_NOMPI`` is set.
See :doc:`../tutorial/parallel_solver` for the details of the two-level parallelization (algorithm layer × solver groups).

- ``setup(nalg=None, nsolve=None, comm=None)`` : Splits the communicators. It must be called before constructing ``Solver`` / ``Algorithm`` (called internally when ``odatse.initialize()`` is used). ``comm`` is the intracommunicator to split; it defaults to ``MPI.COMM_WORLD``. Calling ``setup()`` again with the same effective configuration (the same communicator and the same ``nalg`` / ``nsolve`` after the missing value is derived) does nothing; a different configuration raises ``RuntimeError``. Communicators are compared as MPI handles (mpi4py's ``==``): two Python objects wrapping the same handle are the same communicator, a duplicate (``Dup()``) is not.
- ``ready()`` : Whether ``setup()`` has been called (always ``True`` in the non-MPI stub).
- ``comm()`` / ``size()`` / ``rank()`` : The global communicator and its size and rank. They refer to ``MPI.COMM_WORLD`` before ``setup()``, and to the communicator given to ``setup()`` afterwards.
- ``algcomm()`` / ``algsize()`` / ``algrank()`` : The communicator of the algorithm layer and its size and rank.
- ``solcomm()`` / ``solsize()`` / ``solrank()`` : The communicator of the solver group and its size and rank.
- ``run_on_algorithm()`` : Whether the calling process belongs to the algorithm layer.
- ``enabled()`` : Whether MPI is available (``False`` when ``ODATSE_NOMPI`` is set).

``algcomm()`` , ``solcomm()`` , their size and rank accessors, and ``run_on_algorithm()`` raise ``RuntimeError`` before ``setup()`` . This includes everything built on them: besides ``Solver`` / ``Algorithm``, a ``MeshGrid`` that reads or distributes a mesh file (``MeshGrid(info)``, ``MeshGrid.from_file()``, ``store_file()``, ``do_split()``) also needs ``setup()`` first.

When ODAT-SE is used as a library inside another MPI program, that program may or may not have called ``setup()`` already. Check ``ready()`` and partition a communicator only when needed:

.. code-block:: python

    import odatse.mpi

    if not odatse.mpi.ready():
        odatse.mpi.setup(comm=my_comm)   # my_comm: the ranks ODAT-SE may use

Notes on passing a communicator:

- ``setup()`` is collective over ``comm`` : every rank of ``comm`` must call it with the same arguments. ``nalg * nsolve`` must equal the size of ``comm`` .
- Call ``setup(comm=...)`` *before* loading the input: ``Info.from_file()`` broadcasts the input over ``comm()``, which is ``MPI.COMM_WORLD`` until ``setup()`` has been called, so loading the input first on a subset of the ranks deadlocks. ``odatse.initialize()`` called afterwards keeps the partition made by ``setup(comm=...)`` (see above).
- There is no way to undo ``setup()``: the partition lives for the rest of the process, and a later ``setup()`` with another communicator or layout raises ``RuntimeError``.
- The caller keeps ownership of ``comm`` . ODAT-SE does not free it, and it must stay valid while ODAT-SE is in use.
- A duplicate (``comm.Dup()``) is a different communicator: passing it after ``setup()`` has been called with the original raises ``RuntimeError`` .
- ODAT-SE calls ``MPI_Abort`` on ``comm`` when a solver worker fails outside ``evaluate`` . Depending on the MPI implementation this terminates the whole job, not only the ranks of ``comm`` .
