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
)

# Placeholder for internal hol_light_cc_stubby.

py_proto_library(
    name = "hol_light_py_pb2",
    api_version = 2,
    deps = [":hol_light_proto"],
)

cc_proto_library(
    name = "hol_light_cc_proto",
    deps = [":hol_light_proto"],
)

load("//devtools/clif/python:clif_build_rule.bzl", "pyclif_proto_library")

pyclif_proto_library(
    name = "hol_light_pyclif",
    proto_lib = ":hol_light_proto",
)
