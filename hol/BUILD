cc_library(
    name = "hol",
    srcs = [
        "general.cc",
        "kernel.cc",
        "printer.cc",
        "trace.cc",
    ],
    hdrs = [
        "general.h",
        "kernel.h",
        "printer.h",
        "trace.h",
    ],
    deps = ["@org_tensorflow//tensorflow/core:lib"],
)

cc_binary(
    name = "trace",
    srcs = ["trace_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":hol",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)
