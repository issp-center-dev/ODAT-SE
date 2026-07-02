# Unit tests (pytest). Registered with ctest both serially and, when an MPI
# launcher is available, under mpirun. The test files resolve the source tree
# relative to their own location, so they can be run from any working
# directory. "-p no:cacheprovider" avoids races on .pytest_cache when several
# ranks run concurrently.

add_test(
  NAME unit
  COMMAND ${Python3_EXECUTABLE} -m pytest -v -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}
  )

# mpi4py enables MPI automatically; this entry exercises the no-MPI code path
# by forcing the stub context with ODATSE_NOMPI=1.
add_test(
  NAME unit_nompi
  COMMAND ${Python3_EXECUTABLE} -m pytest -v -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}
  )
set_tests_properties(unit_nompi PROPERTIES ENVIRONMENT "ODATSE_NOMPI=1")

find_program(MPIRUN_EXECUTABLE NAMES mpirun mpiexec)

if(MPIRUN_EXECUTABLE)
  add_test(
    NAME unit_mpi2
    COMMAND ${MPIRUN_EXECUTABLE} -np 2 ${Python3_EXECUTABLE} -m pytest -v -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}
    )
  set_tests_properties(unit_mpi2 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_MCA_rmaps_base_oversubscribe=1"
    )

  add_test(
    NAME unit_mpi4
    COMMAND ${MPIRUN_EXECUTABLE} -np 4 ${Python3_EXECUTABLE} -m pytest -v -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}
    )
  set_tests_properties(unit_mpi4 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_MCA_rmaps_base_oversubscribe=1"
    )
endif()
