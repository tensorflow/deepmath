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
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//third_party/py/numpy",
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
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//third_party/py/numpy",
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
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//third_party/py/numpy",
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
