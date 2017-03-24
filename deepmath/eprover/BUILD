# E prover / first order logic machinery

package(
    default_visibility = ["//visibility:public"],
)

load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
)

tf_proto_library(
    name = "protos",
    srcs = [
        "clause.proto",
        "fast_clause.proto",
        "fol_formula.proto",
        "prover_clause_examples.proto",
        "prover_options.proto",
        "prover_result.proto",
        "prover_task.proto",
        "prover_task_stats.proto",
    ],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)
