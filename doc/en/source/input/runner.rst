==========================================
[``runner``] section
==========================================

This section sets the configuration of ``Runner``, which bridges ``Algorithm`` and ``Solver``.
It has three subsections, ``mapping``, ``limitation``, and ``log`` .

- ``ignore_error``

  Format: Boolean (default: false)

  Description:
  A parameter to specify whether a RuntimeError occuured within the direct problem solver is ignored and the calculation is continued with NaN as the result. Note that only the RuntimeError exceptions are captured.


[``runner.mapping``] section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section defines the mapping from an :math:`N` dimensional parameter searched by ``Algorithm``, :math:`x`, to an :math:`M` dimensional parameter used in ``Solver``, :math:`y` .
In the case of :math:`N \ne M`, the parameter ``dimension`` in ``[solver]`` section should be specified.

In the current version, the affine mapping (linear mapping + translation) :math:`y = Ax+b` is available.

- ``A``

  Format: List of list of float, or a string (default: ``[]``)

  Description:
  :math:`N \times M` matrix :math:`A`. An empty list ``[]`` is a shorthand of an identity matrix.
  If you want to set it by a string, arrange the elements of the matrix separated with spaces and newlines (see the example).


- ``b``

  Format: List of float, or a string (default: ``[]``)

  Description:
  :math:`M` dimensional vector :math:`b`. An empty list ``[]`` is a shorthand of a zero vector.
  If you want to set it by a string, arrange the elements of the vector separated with spaces.

For example, both ::

  A = [[1,1], [0,1]]

and ::

  A = """
  1 1
  0 1
  """

mean

.. math::

  A = \left(
  \begin{matrix}
  1 & 1 \\
  0 & 1
  \end{matrix}
  \right).


[``limitation``] section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section defines the limitation (constraint) in an :math:`N` dimensional parameter searched by ``Algorithm``, :math:`x`, in addition of ``min_list`` and ``max_list``.

In the current version, a linear inequation with the form :math:`Ax+b>0` is available. Specifically, you can apply constraints as follows:

.. math::

  A_{1,1} x_{1} + A_{1,2} x_{2} + &... + A_{1,N} x_{N} + b_{1} > 0\\
  A_{2,1} x_{1} + A_{2,2} x_{2} + &... + A_{2,N} x_{N} + b_{2} > 0\\
  &...\\
  A_{M,1} x_{1} + A_{M,2} x_{2} + &... + A_{M,N} x_{N} + b_{M} > 0

where :math:`M` is the number of constraint equations (arbitrary).

- ``co_a``

  Format: List of list of float, or a string (default: ``[]``)

  Description:
  :math:`M \times N` matrix :math:`A` for the constraint equations.
  The number of rows should be the number of constraints :math:`M`, and the number of columns should be the number of search variables :math:`N`.
  You must define ``co_b`` together with this parameter.

- ``co_b``

  Format: List of float, or a string (default: ``[]``)

  Description:
  :math:`M` dimensional vector :math:`b` for the constraint equations.
  You need to set a column vector with the dimension equal to the number of constraints :math:`M`.
  You must define ``co_a`` together with this parameter.

For example, both ::

  A = [[1,1], [0,1]]

and ::

  A = """
  1 1
  0 1
  """

mean

.. math::

  A = \left(
  \begin{matrix}
  1 & 1 \\
  0 & 1
  \end{matrix}
  \right).

Also, the following examples:

.. code-block:: toml

  co_b = [[0], [-1]]

and

.. code-block:: toml

  co_b = """0 -1"""

and

.. code-block:: toml

  co_b = """
  0
  -1
  """

all represent:

.. math::

  b = \left(
  \begin{matrix}
  0 \\
  -1
  \end{matrix}
  \right)

If neither ``co_a`` nor ``co_b`` is defined, no constraint equation will be applied to the search.


[``log``] section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting parametrs related to logging of solver calls.

- ``filename``

  Format: String (default: "runner.log")

  Description: Name of log file.

- ``interval``

  Format: Integer (default: 0)

  Description:
  The log will be written out every time solver is called ``interval`` times.
  If the value is less than or equal to 0, no log will be written.

- ``write_result``

  Format: Boolean (default: false)

  Description: Whether to record the output from solver.

- ``write_input``

  Format: Boolean (default: false)

  Description: Whether to record the input to solver.
