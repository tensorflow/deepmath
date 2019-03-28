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
    name = "proof_assistant_proto",
    srcs = ["proof_assistant.proto"],
    has_services = 1,
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)
