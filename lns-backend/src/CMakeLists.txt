ggml_add_backend_library(ggml-lns
    ggml-lns.cpp
    lns-ops.cpp
)

# xlnscpp uses a header-inclusion model: #include "xlns16.cpp" with defines
target_compile_definitions(ggml-lns PRIVATE xlns16_table xlns16_alt)

# Point to the xlnscpp source directory so #include "xlns16.cpp" resolves
target_include_directories(ggml-lns PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../xlnscpp)
