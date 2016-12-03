package(default_visibility = ["//visibility:public"])

licenses(["permissive"])  # MIT/X11

cc_library(
    name = "CmdLine",
    srcs = glob(["*.cc"]),
    hdrs = glob([
        "*.hh",
        "*.icc",
    ]),
    copts = ["-fexceptions"],
    deps = ["//zz/Prelude"],
)
