ggml_add_backend_library(ggml-lns
    ggml-lns.cpp
    lns-ops.cpp
)

# xlnscpp uses a header-inclusion model: #include "xlns32.cpp" with defines.
# xlns32 has no `xlns32_table` analogue of xlns16's 65536-entry lookup table (would be too costly);
# F32<->xlns32 conversion goes through real log2/exp2. xlns32_alt enables the
# reduced-branching addition path. xlns32_ideal was intentionally not set
# (table approximation, consistent with a realistic hardware target).
target_compile_definitions(ggml-lns PRIVATE xlns32_alt)

# Point to the xlnscpp submodule so #include "xlns32.cpp" resolves.
# When building inside ggml/llama.cpp, xlnscpp must be available at the path
# below, or you can override with -DXLNSCPP_DIR=/path/to/xlnscpp.
if(NOT DEFINED XLNSCPP_DIR)
    set(XLNSCPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../xlnscpp")
endif()
target_include_directories(ggml-lns PRIVATE ${XLNSCPP_DIR})
