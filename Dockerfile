FROM l.gcr.io/google/bazel:0.24.1
WORKDIR /home

# Dependencies.
RUN apt-get update && apt-get install -y python3-pip python-dev libc-ares-dev
RUN pip3 install h5py six numpy wheel mock pyfarmhash grpcio
RUN pip3 install keras_applications==1.0.6 keras_preprocessing==1.0.5 --no-deps
ENV \
  PYTHON_BIN_PATH=/usr/bin/python3 \
  PYTHON_LIB_PATH=/usr/local/lib/python3.5/dist-packages

# Get repository.
COPY . deepmath/
RUN cd deepmath && git submodule update --init

# Configure tensorflow.
RUN cd deepmath/tensorflow && \
  TF_IGNORE_MAX_BAZEL_VERSION=1 \
  TF_NEED_OPENCL_SYCL=0 \
  TF_NEED_COMPUTECPP=1 \
  TF_NEED_ROCM=0 \
  TF_NEED_CUDA=0 \
  TF_ENABLE_XLA=0 \
  TF_DOWNLOAD_CLANG=0 \
  TF_NEED_MPI=0 \
  TF_SET_ANDROID_WORKSPACE=0 \
  CC_OPT_FLAGS="-march=native -Wno-sign-compare" \
  ./configure

# Build deepmath.
# Note: PYTHON_BIN_PATH and --python_path are both necessary.
RUN cd deepmath && \
  bazel build -c opt //deepmath/deephol:main --define grpc_no_ares=true --python_path=$PYTHON_BIN_PATH

# Make a copy without symlinks.
RUN cp -R -L /home/deepmath/bazel-bin/deepmath/deephol/main.runfiles /home/runfiles

### COPY
FROM python:3
WORKDIR /home
COPY --from=0 /usr/local/lib/python3.5/dist-packages /usr/local/lib/python3.5/dist-packages
COPY --from=0 /home/deepmath/bazel-bin/deepmath/deephol/main .
COPY --from=0 /home/runfiles main.runfiles/
COPY --from=0 /home/deepmath/bazel-bin/deepmath/deephol/main.runfiles_manifest .
ENV \
  PYTHON_BIN_PATH=/usr/bin/python3 \
  PYTHON_LIB_PATH=/usr/local/lib/python3.5/dist-packages
# Set deephol:main as entrypoint.
ENTRYPOINT ["/home/main"]
