# Drop-in replacements of internal-only libraries for open-source version.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "build_data",
    srcs = ["build_data.py"],
    srcs_version = "PY3",
)

py_library(
    name = "compare",
    srcs = ["compare.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "error",
    srcs = ["error.py"],
    srcs_version = "PY3",
)

py_test(
    name = "build_data_test",
    srcs = ["build_data_test.py"],
    # python_version_3
    deps = [
        ":build_data",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "error_test",
    srcs = ["error_test.py"],
    # python_version_3
    deps = [
        ":error",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "recordio_util",
    srcs = ["recordio_util.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "predictions_builder",
    srcs = ["predictions_builder.py"],
    srcs_version = "PY3",
    deps = [
        "//deepmath/deephol:deephol_py_proto",
        "//deepmath/deephol:holparam_predictor",
        "//deepmath/deephol:predictions",
    ],
)

py_library(
    name = "proof_assistant",
    srcs = ["proof_assistant.py"],
    srcs_version = "PY3",
    deps = [
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "//net/grpc/python:grpc",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "proof_assistant_test",
    srcs = ["proof_assistant_test.py"],
    # python_version_3
    deps = [
        ":proof_assistant",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "prover_options_helper",
    srcs = ["prover_options_helper.py"],
    srcs_version = "PY3",
    deps = [
        "//deepmath/deephol:deephol_py_proto",
    ],
)

py_test(
    name = "recordio_util_test",
    srcs = ["recordio_util_test.py"],
    # python_version_3
    deps = [
        ":recordio_util",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "sequence_action_generator",
    srcs = ["sequence_action_generator.py"],
    srcs_version = "PY3",
    deps = [
        "//deepmath/deephol:abstract_action_generator",
        "//deepmath/deephol:deephol_py_proto",
        "//deepmath/deephol:proof_search_tree",
        "//deepmath/proof_assistant:proof_assistant_proto_py",
    ],
)

py_test(
    name = "sequence_action_generator_test",
    srcs = ["sequence_action_generator_test.py"],
    # python_version_3
    deps = [
        ":sequence_action_generator",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "disable_tf2",
    srcs = ["disable_tf2.py"],
    srcs_version = "PY3",
)
