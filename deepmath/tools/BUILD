# Miscellaneous tools and tests

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_test(
    name = "boilerplate_test",
    size = "small",
    srcs = ["boilerplate_test.py"],
    data = ["//deepmath:all_files"],
    srcs_version = "PY2AND3",
    deps = ["@six_archive//:six"],
)

py_binary(
    name = "gen_op_stub",
    srcs = ["gen_op_stub.py"],
    srcs_version = "PY2AND3",
)

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
)
