# Directory for DeepHOL.
#
# Interfaces with ITP and ML models.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//visibility:public"],
)

load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

tf_proto_library(
    name = "deephol_proto",
    srcs = ["deephol.proto"],
    protodeps = ["//deepmath/prover/hol_light:hol_light_proto"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "process_sexp",
    srcs = ["process_sexp.py"],
)

py_library(
    name = "mock_predictions_lib",
    srcs = ["mock_predictions_lib.py"],
    deps = [
        ":predictions",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "predictions",
    srcs = ["predictions.py"],
    deps = [
        ":process_sexp",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "predictions_test",
    size = "medium",
    srcs = ["predictions_test.py"],
    deps = [
        ":mock_predictions_lib",
        ":predictions",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
