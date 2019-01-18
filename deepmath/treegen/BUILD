# Generative models for tree-structured logical forms.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "arith_train",
    srcs = ["arith_train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":arith_model",
        ":arith_utils",
        ":jsonl",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "arith_model",
    srcs = ["arith_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":loom_ops",
        "//third_party/py/numpy",
        "@org_fold//tensorflow_fold/public:loom",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "arith_make_data",
    srcs = ["arith_make_data.py"],
    srcs_version = "PY2AND3",
    deps = [":arith_make_data_lib"],
)

py_library(
    name = "arith_make_data_lib",
    srcs = ["arith_make_data.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":arith_utils",
        ":binary_trees",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "arith_make_data_test",
    srcs = ["arith_make_data_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":arith_make_data_lib",
        ":arith_utils",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "arith_model_test",
    srcs = ["arith_model_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":arith_make_data_lib",
        ":arith_model",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "arith_utils",
    srcs = ["arith_utils.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "binary_trees",
    srcs = ["binary_trees.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "loom_ops",
    srcs = ["loom_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/tensorflow_probability",
        "@org_fold//tensorflow_fold/public:loom",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_test(
    name = "loom_ops_test",
    srcs = ["loom_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":loom_ops",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "cnf_model",
    srcs = ["cnf_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cnf_utils",
        ":loom_ops",
        "//deepmath/util:model_utils",
        "//third_party/py/numpy",
        "@org_fold//tensorflow_fold/public:loom",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "cnf_model_test_lib",
    srcs = ["cnf_model_test_lib.py"],
    data = [
        ":testdata/cnf_metadata.json",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cnf_model",
        "//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "cnf_model_test",
    srcs = ["cnf_model_test.py"],
    shard_count = 21,
    srcs_version = "PY2AND3",
    deps = [
        ":cnf_model",
        ":cnf_model_test_lib",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

# Run this test under -c opt.
py_test(
    name = "cnf_model_big_test",
    srcs = ["cnf_model_big_test.py"],
    data = [
        ":testdata/cnf_metadata.json",
    ],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":cnf_model",
        ":cnf_model_test_lib",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "cnf_utils",
    srcs = ["cnf_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "preprocess_json_clauses",
    srcs = ["preprocess_json_clauses.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "cnf_train",
    srcs = ["cnf_train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cnf_model",
        ":cnf_utils",
        ":jsonl",
        "//deepmath/util:model_utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "jsonl",
    srcs = ["jsonl.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
)
