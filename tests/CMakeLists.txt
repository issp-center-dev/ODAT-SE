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
