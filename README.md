# build-syntaxnet
A package for installing the (good old) SyntaxNet (Parsey McParseface) dependency parser.

## Install Anaconda2:

    $ cd build
    $ wget https://repo.anaconda.com/archive/Anaconda2-5.1.0-Linux-x86_64.sh
    $ cd ..
    $ bash build/Anaconda2-5.1.0-Linux-x86_64.sh

optional: add `anaconda2/bin` to `PATH`

optional: update

    $ conda update conda
    $ conda update anaconda


## Install bazel:

    $ cd build
    $ wget https://github.com/bazelbuild/bazel/releases/download/0.3.1/bazel-0.3.1-installer-linux-x86_64.sh
    $ chmod +x bazel-0.3.1-installer-linux-x86_64.sh
    $ ./bazel-0.3.1-installer-linux-x86_64.sh --user
    $ cd ..

add `$HOME/bin` to `PATH`
  
## Config and install Tensorflow:

    $ cd tensorflow-0.10.0
    $ bash ./configure

If choose GPU support and CUDA is not installed at `/usr/local`, edit file `third_party/gpus/crosstool/CROSSTOOL.tpl`:

    # Include directory for cuda headers.
    # cxx_builtin_include_directory: "/usr/local/cuda%{cuda_version}/include"
    cxx_builtin_include_directory: "/path/to/cuda/include"

Now build the python package:

* with GPU:

    `$ bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package`

* without GPU:

    `$ bazel build -c opt //tensorflow/tools/pip_package:build_pip_package`

Install the python package:

    $ mkdir -p ../build/tensorflow_pkg
    $ bazel-bin/tensorflow/tools/pip_package/build_pip_package ../build/tensorflow_pkg
    $ cd ..
    $ pip install --ignore-installed --upgrade build/tensorflow_pkg/tensorflow-0.10.0-py2-none-any.whl

## Build Syntaxnet:

    $ cd syntaxnet-goodold-repo
    $ bazel build -c opt //syntaxnet:parser_ops.so

## Install Syntaxnet:

    $ cp -r ../syntaxnet ~
    $ cp bazel-bin/syntaxnet/parser_ops.so ~/syntaxnet/
    $ cp -r syntaxnet/models/parsey_mcparseface ~/syntaxnet/
    $ mv ~/syntaxnet/parsey_mcparseface_context.pbtxt ~/syntaxnet/parsey_mcparseface/context.pbtxt
    $ cp ../bin/* ~/bin

## Check:

    $ echo "John loves Mary." | parse -v
