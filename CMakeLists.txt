cmake_minimum_required(VERSION 3.24)
project(salikorn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/std:c++latest /utf-8 /W4 /permissive- /Zc:__cplusplus /EHsc /MP)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

set(SAL_DML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/directml")

# ---------------------------------------------------------------------------
# sal_core — implementations split out of the src/ headers (hpp declares, cpp
# defines). Static lib; each target pulls only the objects whose symbols it uses.
# ---------------------------------------------------------------------------
add_library(sal_core STATIC
  src/gpu/cs.cpp
  src/gpu/autotune.cpp
  src/compress/linalg.cpp
  src/compress/codec_points.cpp
  src/st/safetensors.cpp)
target_include_directories(sal_core PRIVATE src extern extern/eigen)
if(MSVC)
  # /openmp: linalg.cpp's Eigen products were previously inlined into /openmp
  # consumers; keep Eigen's OpenMP path active for identical behavior.
  target_compile_options(sal_core PRIVATE /openmp)
endif()

# ---------------------------------------------------------------------------
# spike_dml — M1/S1+S2: DirectML device probes + FP16 GEMM correctness/perf
# ---------------------------------------------------------------------------
add_executable(spike_dml tools/spikes/spike_dml.cpp)
target_include_directories(spike_dml PRIVATE src "${SAL_DML_DIR}/include")
target_compile_options(spike_dml PRIVATE "/external:I" "${SAL_DML_DIR}/include" /external:W0)
target_link_directories(spike_dml PRIVATE "${SAL_DML_DIR}/bin/x64-win")
target_link_libraries(spike_dml PRIVATE d3d12 dxgi DirectML)
add_custom_command(TARGET spike_dml POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${SAL_DML_DIR}/bin/x64-win/DirectML.dll" "$<TARGET_FILE_DIR:spike_dml>")

# ---------------------------------------------------------------------------
# spike_decomp — E-DECOMP: lossless decomposition experiments (CPU fp64 kit)
# ---------------------------------------------------------------------------
add_executable(spike_decomp tools/spikes/spike_decomp.cpp)
target_include_directories(spike_decomp PRIVATE src extern extern/eigen)
target_link_libraries(spike_decomp PRIVATE sal_core)
if(MSVC)
  target_compile_options(spike_decomp PRIVATE /openmp)
endif()

# ---------------------------------------------------------------------------
# spike_d3d12 — S8: raw compute-shader GEMM on the iGPU (D3DCompile at runtime)
# ---------------------------------------------------------------------------
add_executable(spike_d3d12 tools/spikes/spike_d3d12.cpp)
target_include_directories(spike_d3d12 PRIVATE src)
target_link_libraries(spike_d3d12 PRIVATE d3d12 dxgi)

# ---------------------------------------------------------------------------
# spike_gpu_linalg — GPU analysis linear algebra with precision gates (user dir.)
# ---------------------------------------------------------------------------
add_executable(spike_gpu_linalg tools/spikes/spike_gpu_linalg.cpp)
target_include_directories(spike_gpu_linalg PRIVATE src extern extern/eigen)
target_link_libraries(spike_gpu_linalg PRIVATE sal_core d3d12 dxgi)
if(MSVC)
  target_compile_options(spike_gpu_linalg PRIVATE /openmp)
endif()

# ---------------------------------------------------------------------------
# spike_nonlinear — non-linear vs linear codec bake-off on real weight patches
# ---------------------------------------------------------------------------
add_executable(spike_nonlinear tools/spikes/spike_nonlinear.cpp)
target_include_directories(spike_nonlinear PRIVATE src extern extern/eigen)
target_link_libraries(spike_nonlinear PRIVATE sal_core)
if(MSVC)
  target_compile_options(spike_nonlinear PRIVATE /openmp)
endif()

# ---------------------------------------------------------------------------
# spike_vqvae — VQ-VAE / VAE weight codecs trained on the iGPU
# ---------------------------------------------------------------------------
add_executable(spike_vqvae tools/spikes/spike_vqvae.cpp)
target_include_directories(spike_vqvae PRIVATE src extern extern/eigen)
target_link_libraries(spike_vqvae PRIVATE sal_core d3d12 dxgi)

# ---------------------------------------------------------------------------
# test suite (ctest -L unit | -L gpu | -L perf)
# ---------------------------------------------------------------------------
enable_testing()

function(sal_test name labels)
  add_executable(${name} tests/${name}.cpp)
  target_include_directories(${name} PRIVATE src extern)
  target_link_libraries(${name} PRIVATE sal_core)
  if(MSVC)
    target_compile_options(${name} PRIVATE /openmp)
  endif()
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS "${labels}")
endfunction()

sal_test(test_fp16 "unit")
sal_test(test_safetensors "unit")
sal_test(test_codecs "unit")

add_executable(test_linalg tests/test_linalg.cpp)
target_include_directories(test_linalg PRIVATE src extern extern/eigen)
target_link_libraries(test_linalg PRIVATE sal_core)
if(MSVC)
  target_compile_options(test_linalg PRIVATE /openmp)
endif()
add_test(NAME test_linalg COMMAND test_linalg)
set_tests_properties(test_linalg PROPERTIES LABELS "unit")

add_executable(test_gpu tests/test_gpu.cpp)
target_include_directories(test_gpu PRIVATE src extern extern/eigen)
target_link_libraries(test_gpu PRIVATE sal_core d3d12 dxgi)
if(MSVC)
  target_compile_options(test_gpu PRIVATE /openmp)
endif()
add_test(NAME test_gpu COMMAND test_gpu)
set_tests_properties(test_gpu PROPERTIES LABELS "gpu")

add_executable(autotune tools/autotune.cpp)
target_include_directories(autotune PRIVATE src extern extern/eigen)
target_link_libraries(autotune PRIVATE sal_core d3d12 dxgi)

add_executable(test_autotune tests/test_autotune.cpp)
target_include_directories(test_autotune PRIVATE src extern)
# substitute_tokens/tmpl live in cs.cpp.obj, whose other symbols need the D3D
# import libs; the test itself never touches the GPU at runtime.
target_link_libraries(test_autotune PRIVATE sal_core d3d12 dxgi)
add_test(NAME test_autotune COMMAND test_autotune)
set_tests_properties(test_autotune PROPERTIES LABELS "unit")
