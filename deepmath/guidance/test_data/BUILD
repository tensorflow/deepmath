# See README.md for generation details

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "graphs",
    srcs = [
        "cnn-graph.meta",
        "tree-graph.meta",
    ],
)

filegroup(
    name = "vocabulary",
    srcs = ["vocab"],
)

filegroup(
    name = "vocabulary_large",
    srcs = ["vocab_large"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "examples",
    srcs = glob(["examples-0*"]),
)

filegroup(
    name = "examples_train",
    srcs = glob(["examples-train-0*"]),
)

filegroup(
    name = "examples_eval",
    srcs = glob(["examples-eval-0*"]),
)

filegroup(
    name = "theorems",
    srcs = glob(["theorems/*.txt"]),
)
