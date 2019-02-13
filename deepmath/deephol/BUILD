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
