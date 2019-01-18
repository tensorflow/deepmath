# HOLStep models and utilities

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "parser",
    srcs = ["parser.py"],
    srcs_version = "PY2AND3",
    deps = [":parser_lib"],
)

py_library(
    name = "parser_lib",
    srcs = ["parser.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "parser_test",
    srcs = ["parser_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":parser_lib",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
)
