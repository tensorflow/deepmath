FROM l.gcr.io/google/bazel:latest
WORKDIR /home

# Dependencies.
RUN apt-get update && apt-get install -y python3-pip python-dev libc-ares-dev
RUN pip3 install h5py six numpy wheel mock pyfarmhash grpcio
RUN pip3 install keras_applications==1.0.6 keras_preprocessing==1.0.5 --no-deps
ENV \
  PYTHON_BIN_PATH=/usr/bin/python3\
  PYTHON_LIB_PATH=/usr/local/lib/python3.5/dist-packages

# Get repository.
#RUN git clone https://github.com/tensorflow/deepmath &&\
#  cd deepmath &&\
#  sed -i -e 's/git@github.com:/https:\/\/github.com\//' .gitmodules &&\
#  git submodule update --init
COPY . deepmath/
RUN cd deepmath &&\
  sed -i -e 's/git@github.com:/https:\/\/github.com\//' .gitmodules &&\
  git submodule update --init

# Build tensorflow.
RUN cd deepmath/tensorflow &&\
  TF_IGNORE_MAX_BAZEL_VERSION=1\
  TF_NEED_OPENCL_SYCL=0\
  TF_NEED_COMPUTECPP=1\
  TF_NEED_ROCM=0\
  TF_NEED_CUDA=0\
  TF_ENABLE_XLA=0\
  TF_DOWNLOAD_CLANG=0\
  TF_NEED_MPI=0\
  TF_SET_ANDROID_WORKSPACE=0\
  CC_OPT_FLAGS="-march=native -Wno-sign-compare"\
  ./configure

# It seems that PYTHON_BIN_PATH and --python_path are both necessary.
RUN cd deepmath &&\
  bazel build -c opt //deepmath/deephol:main --define grpc_no_ares=true --python_path=/usr/bin/python3

# Run deephol:main.
ENTRYPOINT /home/deepmath/bazel-bin/deepmath/deephol/main\
  --prover_options=/tmp/example/prover_options.txt\
  --output=/tmp/prooflog.txt\
  --proof_assistant_server_address=hol-light:2000
