# GPU-kmeans
Large scale GPU accelerated k-means algorithm

Could perform the clustering of 600,000 randomly choosen 3D points to 8,000 clusters in less than 4 minutes on my NVIDIA GeForce GTX 1080 Ti with 11,175 MB memory, giving cluster sizes ranging from 39 to 125.

This can be used to anonymize large data sets, by substituting all the points in the data set with the center of the cluster they belong to.

* Writen in Python, with CuPy library
* Automatic mechanism for splitting data into smaller chuncks to fit GPU-card limited memory
* Shows histogram of cluster sizes
* Can save point cloud with colors linked to cluster

Only works with CUDA development kit. You must adapt the version of cupy to your version of CUDA. See [here](https://docs.cupy.dev/en/stable/install.html) for more details on how to install CuPy and GPU libraries from NVIDIA.
