# A C++ HOL Light kernel

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

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
    deps = [
        "//util/gtl:node_hash_map",
        "//util/gtl:node_hash_set",
        "@org_tensorflow//tensorflow/core:lib",
    ],
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

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
)
