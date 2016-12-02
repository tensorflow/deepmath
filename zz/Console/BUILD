package(default_visibility = ["//visibility:public"])

licenses(["permissive"])  # MIT/X11

cc_library(
    name = "Console",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.hh"]),
    deps = ["//zz/Prelude"],
)

cc_binary(
    name = "console_test",
    srcs = ["Main_console.cc"],
    deps = ["Console"],
)
