``Solver``
=========================

``Solver`` is defined as a subclass of ``py2dmat.solver.SolverBase`` ::

    import py2dmat

    class Solver(py2dmat.solver.SolverBase):
        ...

The following methods should be defined.

- ``__init__(self, info: py2dmat.Info)``

    - It is required to call the constructor of the base class.

        - ``super().__init__(info)``    

    - The constructor of ``SolverBase`` defines the following instance variables.

        - ``self.root_dir: pathlib.Path`` : Root directory

            - use ``info.base["root_dir"]``

        - ``self.output_dir: pathlib.Path`` : Output directory

            - use ``info.base["output_dir"]``

        - ``self.proc_dir: pathlib.Path`` : Working directory for each MPI process

            - as ``self.output_dir / str(mpirank)``

        - ``self.work_dir: pathlib.Path`` : Directory where the solver is invoked

            - same to ``self.proc_dir``

    - Read the input parameter ``info`` and save as instance variables.

- ``prepare(self, message: py2dmat.Message) -> None``

    - This is called before the solver starts
    - ``message`` includes an input parameter ``x``, convert it to something to be used by the solver

        - e.g., to generate an input file of the solver

- ``run(self, nprocs: int = 1, nthreads: int = 1) -> None``

    - Run the solver
    - Result should be saved to somewhere in order to be read by ``get_results`` later

        - e.g., save f(x) as an instance variable

- ``get_results(self) -> float``

    - This is called after the solver finishes
    - Returns the result of the solver

        - e.g., to retrieve the result from the output file of the solver
