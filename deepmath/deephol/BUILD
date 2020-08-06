# Directory for DeepHOL.
#
# Interfaces with ITP and ML models.

load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "deephol_proto",
    srcs = ["deephol.proto"],
    protodeps = [
        "//deepmath/proof_assistant:proof_assistant_proto",
        "//deepmath/deephol/utilities:deephol_stat_proto",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "test_data_default_ckpt",
    srcs = glob([
        "test_data/default_ckpt/**",
    ]),
)

filegroup(
    name = "test_data_default_ckpt_training_options",
    srcs = [
        "test_data/default_ckpt/training_options.pbtxt",
    ],
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
    # python_version_3
    deps = [
        ":holparam_predictor",
        ":predictions",
        ":predictions_abstract_base_test",
        ":test_util",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "holparam_predictor",
    srcs = ["holparam_predictor.py"],
    srcs_version = "PY3",
    deps = [
        ":predictions",
        ":process_sexp",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//third_party/py/numpy",
        "//third_party/tensorflow/core/protobuf:config_pyclif_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "process_sexp",
    srcs = ["process_sexp.py"],
    srcs_version = "PY3",
)

py_library(
    name = "mock_predictions",
    srcs = ["mock_predictions.py"],
    data = [
        ":test_data_default_ckpt",
        ":test_data_tac_dep_ckpt",
    ],
    srcs_version = "PY3",
    deps = [
        ":predictions",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "predictions",
    srcs = ["predictions.py"],
    srcs_version = "PY3",
    deps = [
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "predictions_abstract_base_test",
    srcs = ["predictions_abstract_base_test.py"],
    srcs_version = "PY3",
    deps = [
        ":predictions",
        ":theorem_utils",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "proof_assistant_builder",
    srcs = ["proof_assistant_builder.py"],
    srcs_version = "PY3",
    deps = [
        "//deepmath/deephol/google:proof_assistant",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "proof_assistant_builder_test",
    srcs = ["proof_assistant_builder_test.py"],
    # python_version_3
    deps = [
        ":proof_assistant_builder",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY3",
    deps = ["@absl_py//absl/flags"],
)

py_library(
    name = "theorem_fingerprint",
    srcs = ["theorem_fingerprint.py"],
    srcs_version = "PY3",
    deps = ["//deepmath/proof_assistant:proof_assistant_proto_py"],
)

py_test(
    name = "theorem_fingerprint_test",
    srcs = ["theorem_fingerprint_test.py"],
    # python_version_3
    deps = [
        ":theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "to_sexpression",
    srcs = ["to_sexpression.py"],
    srcs_version = "PY3",
    deps = [
        ":predictions",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
    ],
)

py_test(
    name = "to_sexpression_test",
    srcs = ["to_sexpression_test.py"],
    # python_version_3
    deps = [
        ":predictions",
        ":to_sexpression",
        "//deepmath/deephol/utilities:sexpression_parser",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/typing",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "theorem_utils",
    srcs = ["theorem_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "theorem_utils_test",
    srcs = ["theorem_utils_test.py"],
    # python_version_3
    deps = [
        ":theorem_fingerprint",
        ":theorem_utils",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:compare",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_library(
    name = "abstract_action_generator",
    srcs = ["abstract_action_generator.py"],
    srcs_version = "PY3",
    deps = [
        ":deephol_py_proto",
        ":proof_search_tree",
        ":theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "action_generator",
    srcs = ["action_generator.py"],
    srcs_version = "PY3",
    deps = [
        ":abstract_action_generator",
        ":deephol_py_proto",
        ":embedding_store_lib",
        ":predictions",
        ":process_sexp",
        ":proof_search_tree",
        ":theorem_fingerprint",
        ":to_sexpression",
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
    # python_version_3
    deps = [
        ":action_generator",
        ":deephol_py_proto",
        ":embedding_store_lib",
        ":holparam_predictor",
        ":proof_search_tree",
        ":prover_util",
        ":test_util",
        ":theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/typing",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_library(
    name = "embedding_store_lib",
    srcs = ["embedding_store.py"],
    srcs_version = "PY3",
    deps = [
        ":io_util",
        ":predictions",
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
    # python_version_3
    deps = [
        ":embedding_store_lib",
        ":io_util",
        ":mock_predictions",
        ":test_util",
        "//deepmath/deephol/utilities:normalization_lib",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "io_util",
    srcs = ["io_util.py"],
    srcs_version = "PY3",
    deps = [
        ":deephol_py_proto",
        "//deepmath/deephol:theorem_utils",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:recordio_util",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_test(
    name = "io_util_test",
    srcs = ["io_util_test.py"],
    # python_version_3
    deps = [
        ":deephol_py_proto",
        ":io_util",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    # python_version_3
    deps = [
        ":prover_flags",
        ":prover_runner",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "proof_search_tree",
    srcs = ["proof_search_tree.py"],
    srcs_version = "PY3",
    deps = [
        ":deephol_py_proto",
        ":predictions",
        ":tactic_utils",
        ":theorem_fingerprint",
        ":theorem_utils",
        ":to_sexpression",
        "//deepmath/deephol/google:proof_assistant",
        "//deepmath/deephol/utilities:normalization_lib",
        "//deepmath/deephol/utilities:sexpression_parser",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:error",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "proof_search_tree_test",
    srcs = ["proof_search_tree_test.py"],
    # python_version_3
    deps = [
        ":deephol_py_proto",
        ":io_util",
        ":proof_search_tree",
        ":tactic_utils",
        ":theorem_fingerprint",
        ":theorem_utils",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:compare",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/typing",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_library(
    name = "prover",
    srcs = ["prover.py"],
    srcs_version = "PY3",
    deps = [
        ":action_generator",
        ":deephol_py_proto",
        ":proof_search_tree",
        ":prover_util",
        ":prune_lib",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:error",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_library(
    name = "mcts_environment",
    srcs = ["mcts_environment.py"],
    srcs_version = "PY3",
    deps = [
        ":predictions",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "prover_environment",
    srcs = ["prover_environment.py"],
    srcs_version = "PY3",
    deps = [
        ":abstract_action_generator",
        ":action_generator",
        ":deephol_py_proto",
        ":mcts_environment",
        ":predictions",
        ":proof_search_tree",
        "//deepmath/deephol/utilities:deephol_stat_py_proto",
        "//deepmath/deephol/utilities:stats",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "mcts",
    srcs = ["mcts.py"],
    srcs_version = "PY3",
    deps = [
        ":deephol_py_proto",
        ":mcts_environment",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "mcts_prover",
    srcs = ["mcts_prover.py"],
    srcs_version = "PY3",
    deps = [
        ":deephol_py_proto",
        ":mcts",
        ":mcts_environment",
        ":prover",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "mcts_prover_test",
    size = "medium",
    srcs = ["mcts_prover_test.py"],
    # python_version_3
    deps = [
        ":action_generator",
        ":deephol_py_proto",
        ":mcts",
        ":mcts_prover",
        ":mock_predictions",
        ":prover_environment",
        ":prover_util",
        ":theorem_utils",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "prover_flags",
    srcs = ["prover_flags.py"],
    data = [
        ":data/default_prover_options.pbtxt",
    ],
    srcs_version = "PY3",
    deps = [
        ":deephol_py_proto",
        ":io_util",
        ":prover_util",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:prover_options_helper",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "predictions_builder",
    srcs = ["predictions_builder.py"],
    srcs_version = "PY3",
    deps = ["//deepmath/public:predictions_builder"],
)

py_library(
    name = "prover_runner",
    srcs = ["prover_runner.py"],
    srcs_version = "PY3",
    deps = [
        ":action_generator",
        ":deephol_py_proto",
        ":embedding_store_lib",
        ":io_util",
        ":mcts_prover",
        ":predictions",
        ":predictions_builder",
        ":proof_assistant_builder",
        ":prover",
        ":prover_environment",
        "//deepmath/deephol/google:sequence_action_generator",
        "//deepmath/deephol/utilities:stats",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:build_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "prover_util",
    srcs = ["prover_util.py"],
    srcs_version = "PY3",
    deps = [
        ":action_generator",
        ":deephol_py_proto",
        ":io_util",
        ":proof_search_tree",
        ":theorem_fingerprint",
        ":theorem_utils",
        "//deepmath/deephol/utilities:sexpression_parser",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_test(
    name = "prover_util_test",
    size = "medium",
    srcs = ["prover_util_test.py"],
    # python_version_3
    deps = [
        ":action_generator",
        ":proof_search_tree",
        ":prover_util",
        ":theorem_utils",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "prune_lib",
    srcs = ["prune_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":deephol_py_proto",
        ":tactic_utils",
        ":theorem_fingerprint",
        ":theorem_utils",
        "//deepmath/deephol/google:proof_assistant",
        "//deepmath/deephol/utilities:proof_analysis",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:error",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "tactic_utils",
    srcs = ["tactic_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":deephol_py_proto",
        ":theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tactic_utils_test",
    srcs = ["tactic_utils_test.py"],
    data = [
        "//deepmath/deephol:data",
    ],
    # python_version_3
    deps = [
        ":tactic_utils",
        "//deepmath/deephol:deephol_py_proto",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:compare",
        "//third_party/py/typing",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_test(
    name = "abstract_action_generator_test",
    srcs = ["abstract_action_generator_test.py"],
    # python_version_3
    srcs_version = "PY3",
    deps = [
        ":abstract_action_generator",
        ":deephol_py_proto",
        ":proof_search_tree",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
