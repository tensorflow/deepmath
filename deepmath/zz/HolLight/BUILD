package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "HolLight",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.hh"]),
    copts = ["-fexceptions"],
    deps = [
        "//deepmath/zz/CmdLine",
        "//deepmath/zz/Console",
        "//deepmath/zz/Generics",
        "//deepmath/zz/Prelude",
    ],
)

cc_binary(
    name = "pan",
    srcs = ["Main_pan.cc"],
    copts = ["-fexceptions"],
    deps = [
        "HolLight",
        "//deepmath/zz/CmdLine",
        "//deepmath/zz/Prelude",
    ],
)
