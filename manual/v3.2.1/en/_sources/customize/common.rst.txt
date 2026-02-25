Commons
================================

In this section, the components commonly used over the program are described.


``odatse.Info``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This class treats the input parameters.
It contains the following four instance variables.

- ``base`` : ``dict[str, Any]``

  - Parameters for whole program such as the directory where the output will be written.

- ``solver`` : ``dict[str, Any]``

  - Parameters for ``Solver``

- ``algorithm`` : ``dict[str, Any]``

  - Parameters for ``Algorithm``

- ``runner`` : ``dict[str, Any]``

  - Parameters for ``Runner``


An instance of ``Info`` is initialized by passing a ``dict`` which has the following four sub dictionaries, ``base``, ``solver``, ``algorithm``, and ``runner``. (Some of them can be omitted.) 
Each sub dictionary is set to the corresponding field of ``Info``.
Alternatively, it can be created by passing to the class method ``from_file`` a path to input file in TOML format.


``base`` items
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As items of ``base`` field, ``root_dir`` indicating the root dirctory of the calculation, and ``output_dir`` for output results will be set automatically as follows.

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
  - The default value is ``"."``, that is, the same to ``root_dir``
  - Specifically, the following code is executed:

    .. code-block:: python

       p = pathlib.Path(base.get("work_dir", "."))
       p = p.expanduser()
       base["work_dir"] = base["root_dir"] / p


``odatse.Runner``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Runner`` is a class that connects ``Algorithm`` and ``Solver``.
The constructor of ``Runner`` takes instances of ``Solver``, ``Info``, ``Mapping``, and ``Limitation``.
If the instance of ``Mapping`` is omitted, ``TrivialMapping`` is assumed that does no transformation.
If the instance of ``Limitation`` is omitted, ``Unlimited`` is assumed that do not impose constraints.

``submit(self, x: np.ndarray, args: Tuple[int,int]) -> float`` method invokes the solver and returns the value of objective function ``f(x)``.
``submit`` internally uses the instance of ``Limitation`` to check whether the search parameter ``x`` satisfies the constraints. Then, it applies the instance of ``Mapping`` to obtain from ``x`` the input ``y = mapping(x)`` that is actually used by the solver.

See :doc:`../input` for details how/which components of ``info`` ``Runner`` uses.


``odatse.Mapping``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Mapping`` is a class that describes mappings from the search parameters of the inverse problem analysis algorithms to the variables of the direct problem solvers.
It is defined as a function object class that has ``__call__(self, x: np.ndarray) -> np.ndarray`` method.
In the current version, a trivial transformation ``TrivialMapping`` and an affine mapping ``Affine`` are defined.

``TrivialMapping``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TrivialMapping`` provides a trivial transformation :math:`x\to x`, that is, no transformation.
It is taken as a default to the argument of Runner class.

``Affine``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Affine`` provides an affine mapping :math:`x \to y = A x + b`.
The coefficients ``A`` and ``b`` should be given as constructor arguments, or passed as dictionary elements through the ``from_dict`` class method.
In case when they are specified in the input file of ODAT-SE, the format of the parameter may be referred to the input file section of the manual.


``odatse.Limitation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Limitation`` is a class that describes constraints on the :math:`N` dimensional parameter space :math:`x` searched by the inverse problem analysis algorithms.
It is defined as a class that have the method ``judge(self, x: np.ndarray) -> bool``.
In the current version, ``Unlimited`` class that imposes no constraint, and ``Inequality`` class that represents the linear inequality constraint.

``Unlimited``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Unlimited`` represents that no constraint is imposed.
``judge`` method always returns ``True``.
It is taken as a default to the argument of Runner class.


``Inequality``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Inequality`` is a class that expresses :math:`M` constraints imposed on :math:`N` dimensional search parameters :math:`x` in the form :math:`A x + b > 0` where :math:`A` is a :math:`M \times N` matrix and :math:`b` is a :math:`M` dimensional vector.

The coefficients ``A`` and ``b`` should be given as constructor arguments, or passed as dictionary elements through the ``from_dict`` class method.
In case when they are specified in the input file of ODAT-SE, the format of the parameter may be referred to the input file section of the manual.
