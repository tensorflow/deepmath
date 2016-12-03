package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "HolLight",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.hh"]),
    deps = [
        "//zz/CmdLine",
        "//zz/Console",
        "//zz/Generics",
        "//zz/Prelude",
    ],
)

cc_binary(
    name = "pan",
    srcs = ["Main_pan.cc"],
    deps = ["HolLight"],
)
