# DeepHOL utilities (not a dependency of deephol:main).

load(
    "@org_tensorflow//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "deephol_stat_proto",
    srcs = ["deephol_stat.proto"],
    cc_api_version = 2,
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
    # python_version_3
    deps = [
        ":normalization_lib",
        "//deepmath/deephol:theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "proof_analysis",
    srcs = ["proof_analysis.py"],
    deps = [
        "//deepmath/deephol:deephol_py_proto",
        "//deepmath/deephol:proof_search_tree",
        "//deepmath/deephol:to_sexpression",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "proof_analysis_test",
    srcs = ["proof_analysis_test.py"],
    # python_version_3
    deps = [
        ":proof_analysis",
        ":proof_test_util",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "proof_checker",
    srcs = ["proof_checker.py"],
    # python_version_3
    deps = [
        ":proof_checker_lib",
        "//deepmath/deephol:deephol_py_proto",
        "//deepmath/deephol:io_util",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "proof_checker_lib",
    srcs = ["proof_checker_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":proof_analysis",
        "//deepmath/deephol:deephol_py_proto",
        "//deepmath/deephol:tactic_utils",
        "//deepmath/deephol:theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:error",
        "//deepmath/public:proof_assistant",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "proof_checker_test",
    size = "small",
    srcs = ["proof_checker_test.py"],
    # python_version_3
    deps = [
        ":proof_checker_lib",
        "//deepmath/deephol:deephol_py_proto",
        "//deepmath/deephol:tactic_utils",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@protobuf_archive//:protobuf_python",
    ],
)

py_library(
    name = "proof_test_util",
    srcs = ["proof_test_util.py"],
    srcs_version = "PY3",
    deps = [
        "//deepmath/deephol:deephol_py_proto",
        "//deepmath/deephol:theorem_utils",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
    ],
)

py_library(
    name = "sexpression_parser",
    srcs = [
        "sexpression_parser.py",
    ],
    srcs_version = "PY3",
    deps = [],
)

py_test(
    name = "sexpression_parser_test",
    srcs = [
        "sexpression_parser_test.py",
    ],
    # python_version_3
    deps = [
        ":sexpression_parser",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "sexpression_graphs",
    srcs = ["sexpression_graphs.py"],
    srcs_version = "PY3",
    deps = [
        ":sexpression_parser",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "sexpression_graphs_test",
    srcs = ["sexpression_graphs_test.py"],
    # python_version_3
    deps = [
        ":sexpression_graphs",
        "//deepmath/deephol:theorem_utils",
        "//deepmath/deephol:to_sexpression",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "stats",
    srcs = ["stats.py"],
    srcs_version = "PY3",
    deps = [
        ":deephol_stat_py_proto",
        ":proof_analysis",
        "//deepmath/deephol:deephol_py_proto",
        "//deepmath/deephol:io_util",
        "//deepmath/deephol:theorem_fingerprint",
    ],
)

py_test(
    name = "stats_test",
    srcs = ["stats_test.py"],
    # python_version_3
    deps = [
        ":deephol_stat_py_proto",
        ":proof_test_util",
        ":stats",
        "//deepmath/deephol:deephol_py_proto",
        "//deepmath/deephol:theorem_fingerprint",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "//deepmath/public:recordio_util",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "stat_proofs_main",
    srcs = ["stat_proofs_main.py"],
    # python_version_3
    deps = [
        ":stats",
        "//deepmath/deephol:deephol_py_proto",
        "//deepmath/deephol:io_util",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@absl_py//absl/flags",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "sexpression_type_atoms",
    srcs = ["sexpression_type_atoms.py"],
    srcs_version = "PY3",
    deps = [
        ":hol_light_sexpression_trees",
        ":sexpression_parser",
    ],
)

py_test(
    name = "sexpression_type_atoms_test",
    srcs = ["sexpression_type_atoms_test.py"],
    # python_version_3
    deps = [
        ":hol_light_sexpression_syntax",
        ":sexpression_type_atoms",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "hol_light_sexpression_graphs",
    srcs = ["hol_light_sexpression_graphs.py"],
    srcs_version = "PY3",
    deps = [
        ":hol_light_sexpression_syntax",
        ":sexpression_graphs",
        "//third_party/py/enum",
    ],
)

py_test(
    name = "hol_light_sexpression_graphs_test",
    srcs = ["hol_light_sexpression_graphs_test.py"],
    # python_version_3
    deps = [
        ":hol_light_sexpression_graphs",
        ":hol_light_sexpression_syntax",
        ":sexpression_graphs",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "hol_light_sexpression_syntax",
    srcs = ["hol_light_sexpression_syntax.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/enum",
    ],
)

py_library(
    name = "hol_light_sexpression_trees",
    srcs = ["hol_light_sexpression_trees.py"],
    srcs_version = "PY3",
    deps = [
        ":hol_light_sexpression_syntax",
        ":sexpression_parser",
    ],
)

py_test(
    name = "hol_light_sexpression_trees_test",
    srcs = ["hol_light_sexpression_trees_test.py"],
    # python_version_3
    deps = [
        ":hol_light_sexpression_syntax",
        ":hol_light_sexpression_trees",
        ":sexpression_parser",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "sexpression_de_bruijn_indexer",
    srcs = ["sexpression_de_bruijn_indexer.py"],
    srcs_version = "PY3",
    deps = [
        ":hol_light_sexpression_syntax",
        ":hol_light_sexpression_trees",
        ":sexpression_parser",
    ],
)

py_test(
    name = "sexpression_de_bruijn_indexer_test",
    srcs = ["sexpression_de_bruijn_indexer_test.py"],
    # python_version_3
    deps = [
        ":sexpression_de_bruijn_indexer",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
