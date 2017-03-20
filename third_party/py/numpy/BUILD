licenses(["restricted"])  # BSD-3

package(default_visibility = ["//visibility:public"])

py_library(
    name = "numpy",
    srcs = ["numpy_dummy.py"],
    srcs_version = "PY2AND3",
)

genrule(
    name = "dummy",
    outs = ["numpy_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
