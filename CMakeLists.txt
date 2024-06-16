cmake_minimum_required(VERSION 3.10)
project(makemore)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Find the libtorch package
find_package(Torch REQUIRED)

# Find Python and NumPy
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

# Include matplotlib-cpp
include_directories(/opt/matplotlib-cpp/matplotlib-cpp)


# Add source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Add executable
add_executable(makemore ${SOURCES})

# Link libraries
target_link_libraries(makemore "${TORCH_LIBRARIES}" ${Python3_LIBRARIES} ${Python3_NumPy_LIBRARIES})


#build command cmake -DCMAKE_PREFIX_PATH=${LIBTORCH_PATH} ..