# Premise selection experiments

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_definition_cnn",
    srcs = ["model_definition_cnn.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_definition_cnn_flat1",
    srcs = ["model_definition_cnn_flat1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        ":model",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_definition_cnn_flat2",
    srcs = ["model_definition_cnn_flat2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        ":model",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_definition_cnn_flat3",
    srcs = ["model_definition_cnn_flat3.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        ":model",
        "//deepmath/util:model_utils",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_definition_hybrid_pool1",
    srcs = ["model_definition_hybrid_pool1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_definition_hybrid_pool2",
    srcs = ["model_definition_hybrid_pool2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_definition_cnn_bigger",
    srcs = ["model_definition_cnn_bigger.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_final_lstm",
    srcs = ["model_final_lstm.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_final_gru",
    srcs = ["model_final_gru.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_final_cnn_1x",
    srcs = ["model_final_cnn_1x.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        ":model",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_final_cnn_2x",
    srcs = ["model_final_cnn_2x.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        ":model",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_final_cnn_3x",
    srcs = ["model_final_cnn_3x.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        ":model",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model_final_cnn_3x_lstm",
    srcs = ["model_final_cnn_3x_lstm.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":layers",
        ":model",
        "//deepmath/util:model_utils",
        "//third_party/py/keras",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
)
