``analytical`` solver
************************

``analytical`` is a ``Solver`` that computes a predefined benchmark function :math:`f(x)` for evaluating the performance of search algorithms.

Input parameters
~~~~~~~~~~~~~~~~~~~~~~~~

The ``function_name`` parameter in the ``solver`` section specifies the function to use.

- ``function_name``

  Format: string

  Description: Function name. The following functions are available.

  - ``quadratics``

    - Quadratic (sphere) function

      .. math::

          f(\vec{x}) = \sum_{i=1}^N x_i^2

    - Global minimum: :math:`f(\vec{x}^*) = 0` at :math:`\forall_i\, x_i^* = 0`.

  - ``quartics``

    - Quartic function with two global minima

      .. math::

          f(\vec{x}) = \left(\frac{1}{N}\sum_{i=1}^N (x_i - 1)^2\right) \left(\frac{1}{N}\sum_{i=1}^N (x_i + 1)^2\right)

    - Global minima: :math:`f(\vec{x}^*) = 0` at :math:`\forall_i\, x_i^* = 1` and :math:`\forall_i\, x_i^* = -1`.
    - Saddle point: :math:`f(\vec{0}) = 1`.

  - ``ackley``

    - `Ackley function <https://en.wikipedia.org/wiki/Ackley_function>`_

      .. math::

        f(\vec{x}) = 20 + e - 20\exp\!\left[-0.2\sqrt{\frac{1}{N}\sum_{i=1}^N x_i^2}\right] - \exp\!\left[\frac{1}{N}\sum_{i=1}^N\cos(2\pi x_i)\right]

    - Global minimum: :math:`f(\vec{x}^*) = 0` at :math:`\forall_i\, x_i^* = 0`. Has many local minima.

  - ``alpine``

    - Alpine function

      .. math::

          f(\vec{x}) = \sum_{i=1}^N \left|x_i \sin(x_i) + 0.1\, x_i\right|

    - Global minimum: :math:`f(\vec{x}^*) = 0` at :math:`\forall_i\, x_i^* = 0`.

  - ``exponential``

    - Exponential function

      .. math::

          f(\vec{x}) = -\exp\!\left(-\frac{1}{2}\sum_{i=1}^N x_i^2\right)

    - Global minimum: :math:`f(\vec{x}^*) = -1` at :math:`\forall_i\, x_i^* = 0`.

  - ``griewank``

    - Griewank-type function

      .. math::

          f(\vec{x}) = 1 + \frac{1}{4000}\sum_{i=1}^N x_i^2 + \prod_{i=1}^N \cos\!\left(\frac{x_i}{\sqrt{i}}\right)

    - Note: the cosine product term is added (not subtracted), so this differs from the standard Griewank function.

  - ``himmelblau``

    - `Himmelblau's function <https://en.wikipedia.org/wiki/Himmelblau%27s_function>`_ (:math:`N = 2` only)

      .. math::

          f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    - Four global minima: :math:`f = 0` at :math:`(3,\,2)`, :math:`(-2.805118,\,3.131312)`, :math:`(-3.779310,\,-3.283186)`, and :math:`(3.584428,\,-1.848126)`.

  - ``michalewicz``

    - `Michalewicz function <https://en.wikipedia.org/wiki/Michalewicz_function>`_

      .. math::

          f(\vec{x}) = -\sum_{i=1}^N \sin(x_i)\left[\sin\!\left(\frac{i\, x_i^2}{\pi}\right)\right]^{20}

    - The global minimum value and location depend on the dimension. There are :math:`d!` local minima.
      For :math:`d = 2`, :math:`f(\vec{x}^*) \approx -1.8013` at :math:`\vec{x}^* \approx (2.2051,\,1.5698)`.
      For :math:`d = 5`, :math:`f(\vec{x}^*) \approx -4.6876`.
      For :math:`d = 10`, :math:`f(\vec{x}^*) \approx -9.6602`.

  - ``qing``

    - Qing function

      .. math::

          f(\vec{x}) = \sum_{i=1}^N \left(x_i^2 - i\right)^2

    - Global minimum: :math:`f(\vec{x}^*) = 0` at :math:`x_i^* = \pm\sqrt{i}` for :math:`i = 1, \ldots, N`.

  - ``rastrigin``

    - `Rastrigin function <https://en.wikipedia.org/wiki/Rastrigin_function>`_

      .. math::

          f(\vec{x}) = 10N + \sum_{i=1}^N \left[x_i^2 - 10\cos(2\pi x_i)\right]

    - Global minimum: :math:`f(\vec{x}^*) = 0` at :math:`\forall_i\, x_i^* = 0`.

  - ``rosenbrock``

    - `Rosenbrock function <https://en.wikipedia.org/wiki/Rosenbrock_function>`_

      .. math::

          f(\vec{x}) = \sum_{i=1}^{N-1} \left[100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2\right]

    - Global minimum: :math:`f(\vec{x}^*) = 0` at :math:`\forall_i\, x_i^* = 1`.

  - ``schaffer``

    - Schaffer function (generalized)

      .. math::

          f(\vec{x}) = \sum_{i=1}^{N-1} \left[0.5 + \frac{\sin^2\!\left(x_i^2 + x_{i+1}^2\right) - 0.5}{\left(1 + 0.001\,(x_i^2 + x_{i+1}^2)\right)^2}\right]

    - Global minimum: :math:`f(\vec{x}^*) = 0` at :math:`\forall_i\, x_i^* = 0`.

  - ``schwefel``

    - `Schwefel function <https://en.wikipedia.org/wiki/Schwefel_function>`_

      .. math::

          f(\vec{x}) = 418.9829\,N - \sum_{i=1}^N x_i \sin\!\left(\sqrt{|x_i|}\right)

    - Global minimum: :math:`f(\vec{x}^*) \approx 0` at :math:`\forall_i\, x_i^* \approx 420.9687`.

  - ``linear_regression_test``

    - Negative log-likelihood of a linear regression model :math:`y = at + b` with Gaussian noise :math:`\mathcal{N}(0, \sigma^2)`, trained on data :math:`\{(t_k, y_k)\} = \{(1,1),(2,3),(3,2),(4,4),(5,3),(6,5)\}`.
      Parameters: :math:`a = x_1`, :math:`b = x_2`, :math:`\log\sigma^2 = x_3` (:math:`N = 3` only).

      .. math::

          f(a, b, s) = \frac{1}{2}\left[n\,s + e^{-s}\sum_{k=1}^{n}(a\,t_k + b - y_k)^2\right]
          \quad (s = \log\sigma^2,\; n = 6)

    - Global minimum: :math:`f(\vec{x}^*) \approx 1.005071` at :math:`\vec{x}^* \approx (0.628571,\,0.8,\,-0.664976)`.
