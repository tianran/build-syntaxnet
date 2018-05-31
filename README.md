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
    $ . configure

If choose GPU support and CUDA is not installed at `/usr/local`, edit file `third_party/gpus/crosstool/CROSSTOOL.tpl`:

    # Include directory for cuda headers.
    # cxx_builtin_include_directory: "/usr/local/cuda%{cuda_version}/include"
    cxx_builtin_include_directory: "/path/to/cuda/include"

Now build and install the python package:

    $ bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
    $ bazel-bin/tensorflow/tools/pip_package/build_pip_package ../build/tensorflow_pkg
    $ cd ..
    $ pip install --ignore-installed --upgrade build/tensorflow_pkg/tensorflow-0.10.0-py2-none-any.whl

## Build Syntaxnet:

    $ cd syntaxnet-goodold-repo
    $ bazel build -c opt //syntaxnet:parser_ops.so
    $ cp bazel-bin/syntaxnet/parser_ops.so ../syntaxnet
    $ cp -r syntaxnet/models/parsey_mcparseface ../syntaxnet
    $ cd ..

## Install Syntaxnet:

    $ mv bin/* ~/bin
    $ mv syntaxnet ~
    $ mv ~/syntaxnet/parsey_mcparseface_context.pbtxt ~/syntaxnet/parsey_mcparseface/context.pbtxt

## Check:

    $ echo "John loves Mary." | parse -v
