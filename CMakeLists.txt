cmake_minimum_required(VERSION 3.14)
project(NeuralNetFramework)

# Usa C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Incluir directorios de cabecera
include_directories(include)

# Buscar todos los archivos fuente en src/
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

# Ejecutable principal desde src/main.cpp
add_executable(neuralnet ${SOURCES})

# Opcional: compilar ejemplo de MNIST
file(GLOB_RECURSE EXAMPLES CONFIGURE_DEPENDS
    examples/*.cpp
)

foreach(example_file ${EXAMPLES})
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file} ${SOURCES})
endforeach()
