package(default_visibility = ["//visibility:public"])

licenses(["permissive"])  # MIT/X11

cc_library(
    name = "Console",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.hh"]),
    copts = ["-fexceptions"],
    deps = ["//deepmath/zz/Prelude"],
)

cc_binary(
    name = "console_test",
    srcs = ["Main_console.cc"],
    copts = ["-fexceptions"],
    deps = [
        "Console",
        "//deepmath/zz/Prelude",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
)
