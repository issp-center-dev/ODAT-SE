================================
Customization
================================

I want to optimize my own function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :doc:`/customize/tutorial_solver` for a tutorial on defining your own objective function and minimizing it with ODAT-SE.
It provides a complete, copy-paste-ready example.


I want to use an external program as a solver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can call external programs using ``subprocess`` inside the ``evaluate`` method.

.. code-block:: python

    import subprocess

    class MySolver(odatse.solver.SolverBase):
        def evaluate(self, x, args=(), nprocs=1, nthreads=1):
            # Write parameters to file
            with open("params.dat", "w") as f:
                for xi in x:
                    f.write(f"{xi}\n")

            # Run external program
            subprocess.run(["./my_program", "params.dat"], check=True)

            # Read result
            with open("result.dat") as f:
                fx = float(f.read().strip())

            return fx

The ``evaluate`` method is called with ``proc_dir`` (per-process working directory) as the current directory, so there are no file conflicts during MPI parallel execution.

For developing a full-featured package, consider using the solver templates available in `ODAT-SE Gallery <https://isspns-gitlab.issp.u-tokyo.ac.jp/takeohoshi/odat-se-gallery>`_ (see :doc:`/gallery`).


I want to see real analysis examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ODAT-SE Gallery <https://isspns-gitlab.issp.u-tokyo.ac.jp/takeohoshi/odat-se-gallery>`_ provides working examples for TRHEPD, SXRD, LEED, and XAFS analysis.
Each example includes input files, execution scripts, and visualization scripts that can be run as-is.
See :doc:`/gallery` for details.
