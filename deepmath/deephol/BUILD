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
    cc_api_version = 2,
    deps = ["//deepmath/prover/hol_light:hol_light_proto"],
)

py_proto_library(
    name = "deephol_py_pb2",
    api_version = 2,
    deps = [":deephol_proto"],
)

cc_proto_library(
    name = "deephol_cc_proto",
    deps = [":deephol_proto"],
)

load("//devtools/clif/python:clif_build_rule.bzl", "pyclif_proto_library")

pyclif_proto_library(
    name = "deephol_pyclif",
    proto_lib = ":deephol_proto",
)
