``analytical`` solver
************************

``analytical`` is a ``Solver`` that computes a predefined benchmark function :math:`f(x)` for evaluating the performance of search algorithms.

Input parameters
~~~~~~~~~~~~~~~~~~~~~~~~

The ``funtion_name`` parameter in the ``solver`` section specifies the function to use.

- ``function_name``

  Format: string

  Description: Function name. The following functions are available.

  - ``quadratics``

    - Quadratic function

      .. math::

          f(\vec{x}) = \sum_{i=1}^N x_i^2

    - The optimized value :math:`f(\vec{x}^*) = 0 \quad (\forall_i x_i^* = 0)`

  - ``rosenbrock``

    - `Rosenbrock function <https://en.wikipedia.org/wiki/Rosenbrock_function>`_

    .. math::

      f(\vec{x}) = \sum_{i=1}^{N-1} \left[ 100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \right]

    - The optimized value :math:`f(\vec{x}^*) = 0 \quad (\forall_i x_i^* = 1)`

  - ``ackley``

    - `Ackley function  <https://en.wikipedia.org/wiki/Ackley_function>`_

    .. math::

      f(\vec{x}) = 20 + e - 20\exp\left[-0.2\sqrt{\frac{1}{N}\sum_{i=1}^N x_i^2}\right] - \exp\left[\frac{1}{N}\cos\left(2\pi x_i\right)\right]

    - The optimized value :math:`f(\vec{x}^*) = 0 \quad (\forall_i x_i^* = 0)`

  - ``himmerblau``

    - `Himmerblau function <https://en.wikipedia.org/wiki/Himmelblau%27s_function>`_

    .. math::
      
      f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2

    - The optimized value :math:`f(3,2) = f(-2.805118, 3.131312) = f(-3.779310, -3.283186) = f(3.584428, -1.848126) = 0`
