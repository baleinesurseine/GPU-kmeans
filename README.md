# GPU-kmeans
Large scale GPU accelerated k-means algorithm

* Writen in Python, with CuPy library
* Automatic mechanism for splitting data into smaller chuncks to fit GPU-card limited memory
* Shows histogram of cluster sizes
* Can save point cloud with colors linked to cluster

Only works with CUDA development kit. You must adapt the version of cupy to your version of CUDA. See [here](https://docs.cupy.dev/en/stable/install.html) for more details: 
