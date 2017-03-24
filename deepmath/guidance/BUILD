# Proof guidance experiments

package(
    default_visibility = ["//visibility:public"],
)

load("//deepmath:deepmath.bzl", "deepmath_op_library")

licenses(["notice"])  # Apache 2.0

py_library(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//deepmath/util:model_utils",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "train_test",
    srcs = ["train_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":train",
        "//deepmath/util:test_utils",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

cc_library(
    name = "vocabulary",
    srcs = ["vocabulary.cc"],
    hdrs = ["vocabulary.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@protobuf//:protobuf_headers",
    ],
)

cc_test(
    name = "vocabulary_test",
    srcs = ["vocabulary_test.cc"],
    deps = [
        ":vocabulary",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "serialize",
    srcs = ["serialize.cc"],
    hdrs = ["serialize.h"],
    deps = [
        ":vocabulary",
        "//deepmath/eprover:protos_cc",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

cc_library(
    name = "make_fast_clause",
    hdrs = ["make_fast_clause.h"],
    deps = [
        "//deepmath/eprover:protos_cc",
    ],
)

cc_library(
    name = "clause_utils",
    srcs = ["clause_utils.cc"],
    hdrs = ["clause_utils.h"],
    deps = [
        ":vocabulary",
        "//deepmath/eprover:protos_cc",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//third_party/eigen3",
    ],
)

deepmath_op_library(
    name = "clause_ops",
    srcs = [
        "convert_ops.cc",
        "sequence_op.cc",
        "weaver_op.cc",
    ],
    deps = [
        ":clause_utils",
        ":make_fast_clause",
        ":serialize",
        ":vocabulary",
        "//deepmath/eprover:protos_cc",
        "@org_fold//tensorflow_fold/public:loom_cc",
    ],
)

py_library(
    name = "inputs",
    srcs = ["inputs.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_clause_ops",
        ":jagged",
        "//deepmath/util:dfile",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "inputs_test",
    srcs = ["inputs_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//deepmath/eprover:protos_py",
        "//deepmath/guidance:inputs",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "driver_lib",
    srcs = ["driver_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":all_models",
        "//deepmath/guidance:inputs",
        "//deepmath/guidance:jagged",
        "//deepmath/guidance:train",
        "//deepmath/util:model_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "driver_test",
    srcs = ["driver_test.py"],
    data = [
        "//deepmath/guidance/test_data:examples_eval",
        "//deepmath/guidance/test_data:examples_train",
        "//deepmath/guidance/test_data:vocabulary",
    ],
    shard_count = 14,
    srcs_version = "PY2AND3",
    deps = [
        ":driver_lib",
        "//deepmath/eprover:protos_py",
        "//deepmath/util:test_utils",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "driver",
    srcs = ["driver.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":driver_lib",
    ],
)

py_library(
    name = "clause_loom",
    srcs = ["clause_loom.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_clause_ops",
        "@org_fold//tensorflow_fold/public:loom",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "clause_loom_test",
    srcs = ["clause_loom_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":clause_loom",
        "//deepmath/eprover:protos_py",
        "//deepmath/guidance:inputs",
        "//deepmath/util:model_utils",
        "//third_party/py/numpy",
        "@org_fold//tensorflow_fold/public:loom",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "wavenet",
    srcs = ["wavenet.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "all_models",
    srcs = [
        "all_models.py",
        "cnn_unconditional.py",
        "fast_cnn.py",
        "model_cnn_regularized.py",
        "model_wavenet.py",
        "tree_rnn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":clause_loom",
        ":wavenet",
        "//deepmath/guidance:inputs",
        "//deepmath/guidance:jagged",
        "//deepmath/premises:model",
        "//deepmath/premises:model_definition_cnn_flat3",
        "//deepmath/premises:model_final_cnn_3x_lstm",
        "//deepmath/util:model_utils",
        "//third_party/py/keras",
        "@org_fold//tensorflow_fold/public:loom",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "all_models_test",
    srcs = ["all_models_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":all_models",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "inference_graph",
    srcs = ["inference_graph.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":driver_lib",
    ],
)

deepmath_op_library(
    name = "jagged_ops",
    srcs = ["jagged_ops.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core/kernels:bounds_check_lib",
    ],
)

py_library(
    name = "jagged",
    srcs = ["jagged.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_jagged_ops",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "jagged_test",
    srcs = ["jagged_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":jagged",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
)
