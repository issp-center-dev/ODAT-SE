Commons
=========

``py2dmat.Info``
~~~~~~~~~~~~~~~~~~

This class treats the input parameters.
This has the following four instance variables.

- ``base : dict[str, Any]``

    - Parameters for whole program such as the directory where the output will be written.

- ``solver : dict[str, Any]``

    - Parameters for ``Solver``

- ``algorithm : dict[str, Any]``

    - Parameters for ``Algorithm``

- ``runner : dict[str, Any]``

    - Parameters for ``Runner``


An instance of ``Info`` is initialized by passing a ``dict`` which has the following four sub dictionaries, ``base``, ``solver``, ``algorithm``, and ``"runner"``.
Each value will be set to the corresponding field of ``Info``.

- About ``base``

    - Root directory ``root_dir``

        - The default value is ``"."`` (the current directory).
        - Value of ``root_dir`` will be converted to an absolute path.
        - The leading ``~`` will be expanded to the user's home directory.
        - Specifically, the following code is executed ::

            p = pathlib.Path(base.get("root_dir", "."))
            base["root_dir"] = p.expanduser().absolute()

    - Output directory ``output_dir``

        - The default value is ``"."``, that is, the same to ``root_dir``
        - The leading ``~`` will be expanded to the user's home directory.
        - If a relative path is given, its origin is ``root_dir``.
        - Specifically, the following code is executed ::

            p = pathlib.Path(base.get("work_dir", "."))
            p = p.expanduser()
            base["work_dir"] = base["root_dir"] / p


``py2dmat.Message``
~~~~~~~~~~~~~~~~~~~~~~

When ``Algorithm`` tries to invoke ``Solver``, an instance of this class is passed from ``Algorithm`` to ``Solver`` via ``Runner``.

This has the following three instance variables.

- ``x: np.ndarray``

    - Coordinates of a point :math:`x` to calculate :math:`f(x)`

- ``step: int``

    - The index of parameters
    - For example, the index of steps in ``exchange`` and the ID of parameter in ``mapper``.

- ``set: int``

    - Which lap it is
    - For example, ``min_search`` has two laps, the first one is optimization and the second one is recalculation the optimal values for each step.

``py2dmat.Runner``
~~~~~~~~~~~~~~~~~~~~~~~~~~

``Runner`` connects ``Algorithm`` and ``Solver``.
The constructor of ``Runner`` takes ``solver: Solver``, ``info: Info``, and ``mapping: Callable[[np.ndarray], np.ndarray]``.

``submit(self, message: py2dmat.Message) -> float`` method invokes the solver and returns the result.
To evaluate ``fx = f(x)``, use the following code snippet::

    message = py2dmat.Message(x, step, set)
    fx = runner.submit(message)

``submit`` internally uses ``mapping`` for generating a parameter used in ``Solver``, :math:`y`, from a parameter searched by ``Algorithm``, :math:`x`, as ``y = mapping(x)``.
When ``mapping`` is omitted in the constructor (or ``None`` is passed), an affine mapping (``py2dmat.util.mapping.Affine(A,b)``) :math:`y=Ax+b` is used as ``mapping``.
The elements of ``A`` and ``b`` are defined in ``info``.
See :doc:`../input` for details how/which components of ``info`` ``Runner`` uses.
