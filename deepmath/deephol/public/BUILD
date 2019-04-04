# Dependencies used only in open-source version.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

py_library(
    name = "recordio_util",
    srcs = ["recordio_util.py"],
    deps = ["@protobuf_archive//:protobuf_python"],
)

py_library(
    name = "proof_assistant",
    srcs = ["proof_assistant.py"],
    deps = [
        "//deepmath/proof_assistant:proof_assistant_proto_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
