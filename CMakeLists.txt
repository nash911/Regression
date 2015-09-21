#-------CMakeList for Project Regression-----------#

cmake_minimum_required (VERSION 2.6.0)
PROJECT(Regression)

add_executable(Regression
  Source/main.cpp
  Source/dataset.cpp
  Source/regression.cpp
  Source/linear_regression.cpp
  Source/logistic_regression.cpp
)

### executable
target_link_libraries(Regression -g -O2 -larmadillo)

