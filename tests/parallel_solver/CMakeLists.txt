add_test(
  NAME parallel_bayes_grid
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/bayes_grid
  )

add_test(
  NAME parallel_bayes_meshfile
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/bayes_meshfile
  )

add_test(
  NAME parallel_exchange_continuous_multiwalker
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/exchange_continuous
  )

add_test(
  NAME parallel_exchange_continuous_singlewalker
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/exchange_continuous_single
  )

add_test(
  NAME parallel_exchange_grid
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/exchange_grid
  )

add_test(
  NAME parallel_exchange_meshfile
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/exchange_meshfile
  )

add_test(
  NAME parallel_mapper_grid
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/mapper_grid
  )

add_test(
  NAME parallel_mapper_meshfile
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/mapper_meshfile
  )

add_test(
  NAME parallel_minsearch
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/minsearch
  )

add_test(
  NAME parallel_pamc_continuous
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/pamc_continuous
  )

add_test(
  NAME parallel_pamc_continuous_varied
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/pamc_continuous_varied
  )

add_test(
  NAME parallel_pamc_grid
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/pamc_grid
  )

add_test(
  NAME parallel_pamc_meshfile
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/pamc_meshfile
  )

add_test(
  NAME parallel_random_search
  COMMAND sh do.sh
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parallel_solver/random_search
  )

