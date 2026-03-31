ggml_add_backend_library(ggml-lns
    ggml-lns.cpp
    lns-ops.cpp
)

# xlnscpp uses a header-inclusion model: #include "xlns16.cpp" with defines
target_compile_definitions(ggml-lns PRIVATE xlns16_table xlns16_alt)

# Point to the xlnscpp submodule so #include "xlns16.cpp" resolves.
# When building inside ggml/llama.cpp, xlnscpp must be available at the path
# below, or you can override with -DXLNSCPP_DIR=/path/to/xlnscpp.
if(NOT DEFINED XLNSCPP_DIR)
    set(XLNSCPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../xlnscpp")
endif()
target_include_directories(ggml-lns PRIVATE ${XLNSCPP_DIR})
