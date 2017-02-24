# HOLStep models and utilities

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "parser",
    srcs = ["parser.py"],
)

py_test(
    name = "parser_test",
    srcs = ["parser_test.py"],
    deps = [
        ":parser",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
