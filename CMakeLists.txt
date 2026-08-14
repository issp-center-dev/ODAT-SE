cmake_minimum_required(VERSION 3.15)
project(ODAT-SE)
enable_testing()

# Resolve the Python interpreter once and reuse its absolute path in every
# test command. Passing a bare "python3" breaks under mpirun, which re-resolves
# it on each rank to the system interpreter (/usr/bin/python3) that lacks the
# packages installed for the CI/venv Python. LOCATION + VIRTUALENV FIRST make
# CMake pick the same interpreter that a bare "python3" would (active venv, or
# the PATH-prepended CI Python).
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_subdirectory(tests)
