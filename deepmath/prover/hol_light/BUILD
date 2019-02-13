# Encapsulates the HOL Light theorem prover.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//visibility:public"],
)

load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

tf_proto_library(
    name = "hol_light_proto",
    srcs = ["hol_light.proto"],
    has_services = 1,
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)
