if (NOT WITH_SYCL)
    return()
endif ()

core_gather_headers()

gather_srcs(cinnapi_src SRCS
        sycl_module.cc
        sycl_runtime.cc
)