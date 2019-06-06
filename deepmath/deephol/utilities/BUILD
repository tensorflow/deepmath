# DeepHOL utilities (not a dependency of deephol:main).

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//visibility:public"],
)

load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

tf_proto_library(
    name = "deephol_stat_proto",
    srcs = ["deephol_stat.proto"],
    protodeps = ["//deepmath/deephol:deephol_proto"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "normalization_lib",
    srcs = ["normalization_lib.py"],
    deps = [
        ":sexpression_graphs",
        "//deepmath/deephol:theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "normalization_lib_test",
    size = "small",
    srcs = ["normalization_lib_test.py"],
    deps = [
        ":normalization_lib",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "proof_analysis",
    srcs = ["proof_analysis.py"],
    deps = [
        "//deepmath/deephol:deephol_proto_py",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "proof_analysis_test",
    srcs = ["proof_analysis_test.py"],
    deps = [
        ":proof_analysis",
        ":proof_test_util",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "proof_checker",
    srcs = ["proof_checker.py"],
    # python_version_2
    deps = [
        ":proof_checker_lib",
        "//deepmath/deephol:deephol_proto_py",
        "//deepmath/deephol:io_util",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "proof_checker_lib",
    srcs = ["proof_checker_lib.py"],
    deps = [
        ":proof_analysis",
        "//deepmath/deephol:deephol_proto_py",
        "//deepmath/deephol:theorem_fingerprint",
        "//deepmath/deephol/public:proof_assistant",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:error",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "proof_checker_test",
    size = "small",
    srcs = ["proof_checker_test.py"],
    deps = [
        ":proof_checker_lib",
        "//deepmath/deephol:deephol_proto_py",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_library(
    name = "proof_test_util",
    srcs = ["proof_test_util.py"],
    deps = [
        "//deepmath/deephol:deephol_proto_py",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
    ],
)

py_library(
    name = "sexpression_parser",
    srcs = [
        "sexpression_parser.py",
    ],
)

py_test(
    name = "sexpression_parser_test",
    srcs = [
        "sexpression_parser_test.py",
    ],
    deps = [
        ":sexpression_parser",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "sexpression_graphs",
    srcs = ["sexpression_graphs.py"],
    deps = [
        ":sexpression_parser",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "sexpression_graphs_test",
    srcs = ["sexpression_graphs_test.py"],
    deps = [
        ":sexpression_graphs",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "stats",
    srcs = ["stats.py"],
    deps = [
        ":deephol_stat_proto_py",
        ":proof_analysis",
        "//deepmath/deephol:deephol_proto_py",
        "//deepmath/deephol:theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
    ],
)

py_test(
    name = "stats_test",
    srcs = ["stats_test.py"],
    deps = [
        ":deephol_stat_proto_py",
        ":proof_test_util",
        ":stats",
        "//deepmath/deephol:deephol_proto_py",
        "//deepmath/deephol:theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "stat_proofs_main",
    srcs = ["stat_proofs_main.py"],
    # python_version_2
    deps = [
        ":stats",
        "//deepmath/deephol:deephol_proto_py",
        "//deepmath/deephol:io_util",
        "@absl_py//absl/flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
