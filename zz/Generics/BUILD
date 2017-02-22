package(default_visibility = ["//visibility:public"])

licenses(["permissive"])  # MIT/X11

cc_library(
    name = "Generics",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.hh"]),
    copts = ["-fexceptions"],
    deps = ["//zz/Prelude"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
)
