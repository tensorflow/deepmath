# Miscellaneous tools and tests

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_test(
    name = "check_futures_test",
    size = "small",
    srcs = ["check_futures_test.py"],
    data = ["//:all_files"],
    srcs_version = "PY2AND3",
    deps = ["@six_archive//:six"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
)
