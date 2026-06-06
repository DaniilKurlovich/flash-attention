cmake_minimum_required(VERSION 3.16)

project(flash_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(FLASH_ATTENTION_ENABLE_CUDA "Enable future CUDA targets" OFF)

# Development.Module component was added in CMake 3.18.
# For older versions fall back to the Development component.
if(CMAKE_VERSION VERSION_LESS "3.18")
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
else()
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')"
  OUTPUT_VARIABLE TORCH_PYTHON_LIB_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(Torch REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')"
  OUTPUT_VARIABLE PYTHON_EXTENSION_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(FLASH_ATTENTION_SOURCES
  src/csrc/attention_tiled.cpp
)

if(FLASH_ATTENTION_ENABLE_CUDA)
  enable_language(CUDA)
  list(APPEND FLASH_ATTENTION_SOURCES src/csrc/attention_tiled_cuda.cu)
endif()

set(FLASH_ATTENTION_PYTHON_SOURCES
  src/csrc/python_module.cpp
)

set(FLASH_ATTENTION_TEST_SOURCES
  src/csrc/tests/test_attention_tiled.cpp
  src/csrc/tests/test_harness.cpp
  src/csrc/tests/test_main.cpp
)

if(FLASH_ATTENTION_ENABLE_CUDA)
  list(APPEND FLASH_ATTENTION_TEST_SOURCES src/csrc/tests/test_attention_tiled_cuda.cpp)
endif()

function(configure_flash_attention_native_target target_name)
  target_link_libraries(${target_name} PRIVATE ${TORCH_LIBRARIES})
  target_include_directories(
    ${target_name}
    PRIVATE
      ${TORCH_INCLUDE_DIRS}
      ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc
      ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc/tests
  )
  target_compile_definitions(
    ${target_name}
    PRIVATE FLASH_ATTENTION_ENABLE_CUDA=$<BOOL:${FLASH_ATTENTION_ENABLE_CUDA}>
  )

  if(TORCH_CXX_FLAGS)
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(${target_name} PRIVATE ${TORCH_CXX_FLAGS_LIST})
  endif()

  if(FLASH_ATTENTION_ENABLE_CUDA)
    set_target_properties(
      ${target_name}
      PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    target_compile_options(
      ${target_name}
      PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
  endif()
endfunction()

add_library(
  flash_attention_cpp
  MODULE
  ${FLASH_ATTENTION_SOURCES}
  ${FLASH_ATTENTION_PYTHON_SOURCES}
)
configure_flash_attention_native_target(flash_attention_cpp)
target_link_libraries(flash_attention_cpp PRIVATE Python::Module)

set(TORCH_PYTHON_LIBRARY "${TORCH_PYTHON_LIB_DIR}/libtorch_python.so")
if(EXISTS "${TORCH_PYTHON_LIBRARY}")
  target_link_libraries(flash_attention_cpp PRIVATE "${TORCH_PYTHON_LIBRARY}")
else()
  message(WARNING "libtorch_python.so not found at ${TORCH_PYTHON_LIBRARY}")
endif()

set_target_properties(
  flash_attention_cpp
  PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_EXTENSION_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    BUILD_RPATH "${TORCH_PYTHON_LIB_DIR}"
)

add_executable(
  flash_attention_csrc_tests
  ${FLASH_ATTENTION_SOURCES}
  ${FLASH_ATTENTION_TEST_SOURCES}
)
configure_flash_attention_native_target(flash_attention_csrc_tests)

# The test binary links against torch internals that may pull in a newer
# cudart than the system CUDA toolkit provides.
# Add the torch lib directory (and the nvidia/cuda_runtime subdir) to RPATH
# so the loader can resolve libcudart.so.12 at runtime.
if(FLASH_ATTENTION_ENABLE_CUDA AND EXISTS "${TORCH_PYTHON_LIB_DIR}")
  get_filename_component(TORCH_CUDA_RUNTIME_DIR
    "${TORCH_PYTHON_LIB_DIR}/../../nvidia/cuda_runtime/lib"
    ABSOLUTE
  )
  set(TEST_RPATH "${TORCH_PYTHON_LIB_DIR}")
  if(EXISTS "${TORCH_CUDA_RUNTIME_DIR}/libcudart.so.12")
    list(APPEND TEST_RPATH "${TORCH_CUDA_RUNTIME_DIR}")
  endif()
  set_target_properties(
    flash_attention_csrc_tests
    PROPERTIES
      BUILD_RPATH "${TEST_RPATH}"
  )
  # Also link the torch python library so pybind11 symbols resolve.
  if(EXISTS "${TORCH_PYTHON_LIBRARY}")
    target_link_libraries(flash_attention_csrc_tests PRIVATE "${TORCH_PYTHON_LIBRARY}")
  endif()
  # Force the linker to prefer the newer libcudart.so.12 shipped with torch
  # over the system /usr/lib/x86_64-linux-gnu/libcudart.so.
  if(EXISTS "${TORCH_CUDA_RUNTIME_DIR}/libcudart.so.12")
    target_link_libraries(flash_attention_csrc_tests PRIVATE "${TORCH_CUDA_RUNTIME_DIR}/libcudart.so.12")
  endif()
endif()

if(FLASH_ATTENTION_ENABLE_CUDA)
  message(STATUS "CUDA is enabled with a kernel stub in src/csrc/attention_tiled_cuda.cu.")
endif()
