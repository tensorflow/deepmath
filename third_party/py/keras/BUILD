licenses(["notice"])  # MIT

package(default_visibility = ["//visibility:public"])

py_library(
    name = "keras",
    srcs = ["keras_dummy.py"],
    srcs_version = "PY2AND3",
)

genrule(
    name = "dummy",
    outs = ["keras_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
