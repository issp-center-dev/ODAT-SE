add_test(
  NAME bayes
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/bayes
  )

add_test(
  NAME exchange
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/exchange
  )

add_test(
  NAME exchange_mesh
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/exchange_mesh
  )

add_test(
  NAME mapper
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/mapper
  )

add_test(
  NAME mapper_random
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/mapper_random
  )

add_test(
  NAME minsearch
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/minsearch
  )

add_test(
  NAME pamc
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/pamc
  )

add_test(
  NAME pamc_varied
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/pamc_varied
  )

add_test(
  NAME transform
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/transform
  )

add_test(
  NAME bayes_continue
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/bayes_continue
  )

add_test(
  NAME exchange_continue
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/exchange_continue
  )

add_test(
  NAME mapper_resume
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/mapper_resume
  )

add_test(
  NAME pamc_continue
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/pamc_continue
  )

add_test(
  NAME pamc_varied_continue
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/pamc_varied_continue
  )

add_test(
  NAME ttopt
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/ttopt
  )

add_test(
  NAME ttopt_resume
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/ttopt_resume
  )

# The do.sh scripts run "mpirun/mpiexec ... ${PYTHON:-python3}". Point PYTHON at
# the interpreter CMake resolved (see top-level CMakeLists.txt) so the MPI
# launcher does not fall back to the system python3, which lacks the installed
# packages. Applies to the do.sh tests added directly in this directory; the
# unit tests use ${Python3_EXECUTABLE} directly.
get_property(_dosh_tests DIRECTORY PROPERTY TESTS)
if(_dosh_tests)
  set_tests_properties(${_dosh_tests} PROPERTIES ENVIRONMENT "PYTHON=${Python3_EXECUTABLE}")
endif()

add_subdirectory(parallel_solver)

add_subdirectory(unit)

