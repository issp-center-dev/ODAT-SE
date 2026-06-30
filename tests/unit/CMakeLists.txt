# Unit tests (pytest). Registered with ctest both serially and, when an MPI
# launcher is available, under mpirun. The test files resolve the source tree
# relative to their own location, so they can be run from any working
# directory. "-p no:cacheprovider" avoids races on .pytest_cache when several
# ranks run concurrently.

add_test(
  NAME unit
  COMMAND python3 -m pytest -v -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}
  )

find_program(MPIRUN_EXECUTABLE NAMES mpirun mpiexec)

if(MPIRUN_EXECUTABLE)
  add_test(
    NAME unit_mpi2
    COMMAND ${MPIRUN_EXECUTABLE} -np 2 python3 -m pytest -v -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}
    )
  set_tests_properties(unit_mpi2 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_MCA_rmaps_base_oversubscribe=1"
    )

  add_test(
    NAME unit_mpi4
    COMMAND ${MPIRUN_EXECUTABLE} -np 4 python3 -m pytest -v -p no:cacheprovider ${CMAKE_CURRENT_SOURCE_DIR}
    )
  set_tests_properties(unit_mpi4 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OMPI_MCA_rmaps_base_oversubscribe=1"
    )
endif()
