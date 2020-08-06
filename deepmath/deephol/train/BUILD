# Code for interacting with the Hol Light Tactics dataset.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "data",
    srcs = ["data.py"],
    srcs_version = "PY3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "data_test",
    srcs = ["data_test.py"],
    # python_version_3
    deps = [
        ":data",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/tensorflow/contrib/lookup:lookup_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    # python_version_3
    deps = [
        ":utils",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "extractor",
    srcs = ["extractor.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "extractor_test",
    srcs = ["extractor_test.py"],
    # python_version_3
    deps = [
        ":extractor",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "architectures",
    srcs = ["architectures.py"],
    srcs_version = "PY3",
    deps = [
        ":losses",
        ":utils",
        ":wavenet",
        "//third_party/tensorflow/contrib/util:util_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "architectures_test",
    srcs = ["architectures_test.py"],
    # python_version_3
    deps = [
        ":architectures",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/typing",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "losses_test",
    srcs = ["losses_test.py"],
    # python_version_3
    deps = [
        ":losses",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
    deps = [
        ":extractor",
        ":utils",
        "//third_party/tensorflow/contrib/opt:opt_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    # python_version_3
    deps = [
        ":model",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "experiment_lib",
    srcs = ["experiments.py"],
    srcs_version = "PY3",
    deps = [
        ":architectures",
        ":data",
        ":model",
        ":utils",
        "//third_party/py/numpy",
        "//third_party/tensorflow/contrib/estimator:estimator_py",
        "//third_party/tensorflow/contrib/training:training_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "experiment_lib_test",
    srcs = ["experiment_lib_test.py"],
    # python_version_3
    deps = [
        ":experiment_lib",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "experiments",
    srcs = ["experiments.py"],
    # python_version_3
    deps = [
        ":experiment_lib",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ],
)

py_library(
    name = "wavenet",
    srcs = ["wavenet.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/tensorflow/contrib/layers:layers_py",
        "//third_party/tensorflow/contrib/slim",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "wavenet_test",
    srcs = ["wavenet_test.py"],
    # python_version_3
    deps = [
        ":wavenet",
        "//deepmath/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "translate",
    srcs = ["translate.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/language/bert:modeling",
        "//third_party/py/language/bert:optimization",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow_probability",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)
