licenses(["restricted"])  # BSD-3

package(default_visibility = ["//visibility:public"])

py_library(
    name = "scipy",
    srcs = ["scipy_dummy.py"],
    srcs_version = "PY2AND3",
)

genrule(
    name = "dummy",
    outs = ["scipy_dummy.py"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
)
