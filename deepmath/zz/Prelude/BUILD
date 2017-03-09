package(default_visibility = ["//visibility:public"])

licenses(["permissive"])  # MIT/X11

cc_library(
    name = "Prelude",
    srcs = ["Prelude.cc"],
    hdrs = glob([
        "Prelude.hh",
        "*.ihh",
        "*.icc",
    ]),
    copts = ["-fexceptions"],
    defines = [
        "__BAZEL_BUILD__",
        "ZZ_Prelude_hh=\\\"deepmath/zz/Prelude/Prelude.hh\\\"",
        "ZZ_PTHREADS",
        #"ZZ_USE_STL",   # -- uncomment to use 'std::vector' as implementation of 'Vec' (uses ~50% more memory)
        #"ZZ_BIG_MODE",  # -- uncomment to enable >4B Types/Terms/Theorems or proof-steps (almost doubles memory consumption)
    ],
    linkopts = ["-lz"],
)

cc_binary(
    name = "prelude_test",
    srcs = ["Main_prelude_test.cc"],
    copts = ["-fexceptions"],
    deps = ["Prelude"],
)

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
)
