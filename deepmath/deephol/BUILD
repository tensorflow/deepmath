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
    protodeps = ["//deepmath/proof_assistant:proof_assistant_proto"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "test_data_default_ckpt",
    srcs = glob([
        "test_data/default_ckpt/**",
    ]),
)

filegroup(
    name = "test_data_tac_dep_ckpt",
    srcs = glob([
        "test_data/tac_dep_ckpt/**",
    ]),
)

filegroup(
    name = "data",
    srcs = glob([
        "data/**",
    ]),
)

py_test(
    name = "holparam_predictor_test",
    srcs = ["holparam_predictor_test.py"],
    data = [
        ":test_data_default_ckpt",
        ":test_data_tac_dep_ckpt",
    ],
    deps = [
        ":holparam_predictor",
        ":predictions_abstract_base_test",
        ":test_util",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "holparam_predictor",
    srcs = ["holparam_predictor.py"],
    deps = [
        ":predictions",
        ":process_sexp",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
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
    deps = ["//third_party/py/numpy"],
)

py_test(
    name = "predictions_test",
    size = "medium",
    srcs = ["predictions_test.py"],
    deps = [
        ":mock_predictions_lib",
        ":predictions",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "predictions_abstract_base_test",
    srcs = ["predictions_abstract_base_test.py"],
    deps = [
        ":predictions",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = ["@absl_py//absl/flags"],
)

py_library(
    name = "theorem_fingerprint",
    srcs = ["theorem_fingerprint.py"],
)

py_test(
    name = "theorem_fingerprint_test",
    srcs = ["theorem_fingerprint_test.py"],
    deps = [
        ":theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "action_generator",
    srcs = ["action_generator.py"],
    deps = [
        ":deephol_proto_py",
        ":embedding_store_lib",
        ":predictions",
        ":process_sexp",
        ":proof_search_tree",
        ":theorem_fingerprint",
        "//deepmath/deephol/utilities:normalization_lib",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "action_generator_test",
    srcs = ["action_generator_test.py"],
    data = [
        "data/hollight_tactics.textpb",
        ":test_data_default_ckpt",
    ],
    deps = [
        ":action_generator",
        ":deephol_proto_py",
        ":embedding_store_lib",
        ":holparam_predictor",
        ":proof_search_tree",
        ":prover_util",
        ":test_util",
        ":theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_library(
    name = "embedding_store_lib",
    srcs = ["embedding_store.py"],
    deps = [
        ":io_util",
        ":predictions",
        ":process_sexp",
        "//deepmath/deephol/utilities:normalization_lib",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "embedding_store_test",
    size = "medium",
    srcs = ["embedding_store_test.py"],
    data = [
        "data/mini_theorem_database.textpb",
    ],
    deps = [
        ":embedding_store_lib",
        ":io_util",
        ":mock_predictions_lib",
        ":process_sexp",
        ":test_util",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "io_util",
    srcs = ["io_util.py"],
    deps = [
        ":deephol_proto_py",
        "//deepmath/deephol/public:recordio_util",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_test(
    name = "io_util_test",
    srcs = ["io_util_test.py"],
    deps = [
        ":deephol_proto_py",
        ":io_util",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    # python_version_2
    deps = [
        ":prover_flags",
        ":prover_runner",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "proof_search_tree",
    srcs = ["proof_search_tree.py"],
    deps = [
        ":deephol_proto_py",
        ":theorem_fingerprint",
        "//deepmath/deephol/public:proof_assistant",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:error",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "prover",
    srcs = ["prover.py"],
    deps = [
        ":action_generator",
        ":deephol_proto_py",
        ":embedding_store_lib",
        ":holparam_predictor",
        ":io_util",
        ":predictions",
        ":proof_search_tree",
        ":prover_util",
        ":prune_lib",
        "//deepmath/deephol/public:proof_assistant",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:error",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_library(
    name = "prover_flags",
    srcs = ["prover_flags.py"],
    deps = [
        ":deephol_proto_py",
        ":io_util",
        ":prover_util",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_library(
    name = "prover_runner",
    srcs = ["prover_runner.py"],
    deps = [
        ":deephol_proto_py",
        ":io_util",
        ":prover",
        "//deepmath/deephol/utilities:stats",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:build_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "prover_util",
    srcs = ["prover_util.py"],
    deps = [
        ":action_generator",
        ":deephol_proto_py",
        ":io_util",
        ":proof_search_tree",
        ":theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_test(
    name = "prover_util_test",
    size = "medium",
    srcs = ["prover_util_test.py"],
    deps = [
        ":action_generator",
        ":proof_search_tree",
        ":prover_util",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "prune_lib",
    srcs = ["prune_lib.py"],
    deps = [
        ":deephol_proto_py",
        ":prover_util",
        "//deepmath/deephol/public:proof_assistant",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:error",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
