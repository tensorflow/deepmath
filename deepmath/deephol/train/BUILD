# Code for interacting with the Hol Light Tactics dataset.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "data",
    srcs = ["data.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "extractor",
    srcs = ["extractor.py"],
    deps = [
        ":utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "architectures",
    srcs = ["architectures.py"],
    deps = [
        ":losses",
        ":utils",
        ":wavenet",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":extractor",
        ":utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "experiment_lib",
    srcs = ["experiments.py"],
    deps = [
        ":architectures",
        ":data",
        ":model",
        ":utils",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "experiments",
    srcs = ["experiments.py"],
    # python_version_2
    deps = [":experiment_lib"],
)

py_library(
    name = "wavenet",
    srcs = ["wavenet.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
