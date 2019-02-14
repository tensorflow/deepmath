# Encapsulates the HOL Light theorem prover.

load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "hol_light_proto",
    srcs = ["hol_light.proto"],
    has_services = 1,
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

py_library(
    name = "theorem_fingerprint",
    srcs = ["theorem_fingerprint.py"],
)

py_test(
    name = "theorem_fingerprint_test",
    srcs = ["theorem_fingerprint_test.py"],
    deps = [
        ":hol_light_proto_py",
        ":theorem_fingerprint",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "error",
    srcs = ["error.py"],
)

py_library(
    name = "hol_light",
    srcs = ["hol_light.py"],
    deps = [
        ":error",
        ":hol_light_proto_py",
    ],
)
