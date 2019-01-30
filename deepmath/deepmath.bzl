"""Build rules for deepmath."""

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)

is_bazel = not hasattr(native, "genmpm")

def deepmath_op_library(name, srcs, deps = []):
    gen = "gen_" + name
    gen_py = gen + ".py"
    if is_bazel:
        so = name + ".so"
        tf_custom_op_library(
            name = so,
            srcs = srcs,
            deps = deps,
        )
        native.genrule(
            name = gen + "_py",
            srcs = [],
            outs = [gen_py],
            cmd = "$(location //deepmath/tools:gen_op_stub) %s > \"$@\"" % name,
            tools = ["//deepmath/tools:gen_op_stub"],
        )
        native.py_library(
            name = gen,
            srcs = [gen_py],
            srcs_version = "PY2AND3",
            data = [":" + so],
            deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
        )
    else:
        extra = [
            "//third_party/tensorflow/core:framework",
            "//third_party/tensorflow/core:lib",
            "//third_party/eigen3",
        ]
        tf_kernel_library(
            name = name,
            srcs = srcs,
            deps = deps + extra,
        )
        wrapper = "gen_" + name + "_impl"
        tf_gen_op_wrapper_py(
            name = gen,
            generated_target_name = wrapper,
            out = gen_py,
            deps = [":" + name],
        )
        native.py_library(
            name = gen,
            deps = [":" + name, ":" + wrapper],
        )
