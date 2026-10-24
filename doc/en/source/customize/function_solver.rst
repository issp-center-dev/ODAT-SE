.. _customize_function_solver:

Defining a solver with ``function.Solver``
==========================================

If the objective function can be written as a Python function ``f(x) -> float``,
``odatse.solver.function.Solver`` lets you create a solver without implementing
the ``evaluate`` method, instead of inheriting ``SolverBase`` directly.
The built-in ``analytical`` solver (:doc:`../solver/analytical`) is also a subclass of this class.

Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

Create an instance and pass the objective function with ``set_function()``.

.. code-block:: python

    import numpy as np
    import odatse
    import odatse.solver.function

    def f(x: np.ndarray) -> float:
        return np.sum((x - 1.0) ** 2)

    solver = odatse.solver.function.Solver(info)
    solver.set_function(f)

``sample/user_function/simple.py`` shows an example that runs a search with this solver.

You can also create a subclass and set the function in its constructor.

.. code-block:: python

    class Solver(odatse.solver.function.Solver):
        def __init__(self, info: odatse.Info):
            super().__init__(info)
            self._name = "my_function"
            self.set_function(f)

Methods and attributes
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``set_function(self, f) -> None``

  Sets the objective function ``f(x: np.ndarray) -> float``.
  Any previously set gradient is cleared. To use a gradient, call ``set_gradient()`` after this.

- ``set_gradient(self, g) -> None``

  Sets the gradient of the objective function ``g(x: np.ndarray) -> np.ndarray``.
  ``g`` must return an array with the same shape as ``x``.
  Once a gradient is set, ``supports_gradient`` becomes True automatically,
  and the gradient can be used, e.g., with ``use_gradient = true`` in ``minsearch``.

- ``evaluate(self, x, args=()) -> float``

  Calls ``f(x)`` with ``work_dir`` as the current directory and returns its value.
  ``args`` is not passed to ``f``.
  Raises ``RuntimeError`` if no function is set.

- ``gradient(self, x, args=()) -> np.ndarray``

  Calls ``g(x)`` and returns its value. ``args`` is not passed to ``g``.
  Unlike ``evaluate``, it does not change the current directory and does not apply ``delay``.
  Raises ``NotImplementedError`` if no gradient is set.

- ``supports_gradient``

  True when a gradient is set.
  A value assigned to ``self.supports_gradient`` in a subclass takes precedence
  and is kept across later calls to ``set_function()`` and ``set_gradient()``.

Input parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

The following parameter can be specified in the ``[solver]`` section.

- ``delay``

  Format: Float (default: 0.0)

  Description: A parameter for debugging. After each function evaluation in ``evaluate``, waits for this number of seconds.
