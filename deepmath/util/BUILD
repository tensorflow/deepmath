# Miscellaneous deepmath utilities

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "model_utils",
    srcs = ["model_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/numpy",
        "@org_fold//tensorflow_fold/public:loom",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "model_utils_test",
    srcs = ["model_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model_utils",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "dfile",
    srcs = ["dfile.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "dfile_test",
    srcs = ["dfile_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":dfile",
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY2AND3",
)

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
)
