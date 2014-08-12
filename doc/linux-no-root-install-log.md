## Install Log

This is a log of my experience setting up Caffe on a cluster running
Redhat 6.3 (Santiago) **without having root**, i.e. by compiling and
installing the necessary dependencies locally.  Basically, I followed
these two sets of instructions but made the appropriate modifications
for my setup:

- http://caffe.berkeleyvision.org/installation.html
- https://github.com/BVLC/caffe/wiki/Installation-%28rbg%29

Notes:

- Here we use Redhat 6.3 instead of Ubuntu 12.04, as assumed by the second link above.
- In this cluster environment, I was lucky and Cuda 5.5 and the Intel toolkit (14.0.0) were already installed.
- I had a checkout of caffe at `$HOME/s/caffe`. If your checkout is at a different location, just change any instances of `s/caffe` to the correct path.



## Satisfy prerequisites

I got lucky and Cuda and Intel MKL were available on my cluster already as Lmod modules, allowing them to be easily loaded:

    module load cuda/5.5
    module load intel/14.0.0

Modify Makefile.config to point to the appropriate locations. Next, install the following series of prerequisites. I assume you can download and untar the source code, so I've left those lines off.

### Install Protobuf

Download from https://code.google.com/p/protobuf/downloads/list

    [~/temp/protobuf-2.5.0] $ ./configure --prefix=$HOME/local
    [~/temp/protobuf-2.5.0] $ make
    [~/temp/protobuf-2.5.0] $ make install

### Install google-glog

    [~/temp/glog-0.3.3] $ ./configure --prefix=$HOME/local
    [~/temp/glog-0.3.3] $ make -j 32
    [~/temp/glog-0.3.3] $ make install
    
### Install snappy

    [~/temp/snappy-1.1.1] $ ./configure --prefix=$HOME/local
    [~/temp/snappy-1.1.1] $ make
    [~/temp/snappy-1.1.1] $ make install
    

### Install leveldb

This one is a little more tricky, because it doesn't follow the usual configure/make/make install pattern, so we have to install the lib and include files manually:

    [~/temp/leveldb-1.15.0] $ make
    [~/temp/leveldb-1.15.0] $ cp -av libleveldb.* $HOME/local/lib/
    [~/temp/leveldb-1.15.0] $ cp -av include/leveldb $HOME/local/include/

### Install OpenCV

    [~/temp] $ wget 'https://github.com/Itseez/opencv/archive/2.4.8.tar.gz'
    [~/temp] $ tar xzf 2.4.8
    [~/temp] $ cd opencv-2.4.8/
    [~/temp/opencv-2.4.8/release] $ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=$HOME/local -D BUILD_opencv_gpu=OFF ..
    [~/temp/opencv-2.4.8/release] $ make
    [~/temp/opencv-2.4.8/release] $ make install

### Install Boost

    [~/temp/boost_1_55_0] $ ./bootstrap.sh --prefix=$HOME/local
    [~/temp/boost_1_55_0] $ ./b2 -j 32
    [~/temp/boost_1_55_0] $ ./b2 install

### Install gflags

    [~/temp] $ git clone https://code.google.com/p/gflags/
    [~/temp/gflags] $ cmake -D CMAKE_INSTALL_PREFIX=$HOME/local .
    [~/temp/gflags] $ make
    [~/temp/gflags] $ make install

### Install lmdb


    [~/temp] $ git clone https://gitorious.org/mdb/mdb.git
    [~/temp/mdb/libraries/liblmdb] $ make
    [~/temp/mdb/libraries/liblmdb] $ make prefix=$HOME/local install


## Build caffe

After install the above prereqs, caffe should compile!

    [~/s/caffe] $ make
    [~/s/caffe] $ ls build/tools
    compute_image_mean.bin*  convert_imageset.o  dump_network.bin*  finetune_net.o            test_net.bin*   train_net.o
    compute_image_mean.o     device_query.bin*   dump_network.o     net_speed_benchmark.bin*  test_net.o
    convert_imageset.bin*    device_query.o      finetune_net.bin*  net_speed_benchmark.o     train_net.bin*
