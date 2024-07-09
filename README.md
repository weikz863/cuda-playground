# CUDA Playground

This is the CUDA playground for PPCA AI project. In this optional part, you will learn how to use CUDA to accelerate your code.

## Setup

We highly recommend to use the docker image provided by NVIDIA. You can use the following command to attach to the docker image in the lab server.

```bash
ssh root@IP -p 2233
```

The IP address and password is the same as the one you use to connect to the server. Note that all the students share the same server, so please be nice and do not destroy the environment.

After you successfully connect to the server, you can clone the repository and run the example code.

```bash
git clone https://github.com/Conless/cuda-playground
cd cuda-playground
make
./build/basic
```

If you see the following output, then you have successfully set up the environment.

```
Hello from thread 0
Hello from thread 1
Hello from thread 2
Hello from thread 3
Hello from thread 4
Hello from thread 5
Hello from thread 6
Hello from thread 7
```

## Task
### Week 1

Learn the basic concept of CUDA and implement the following files.
- `src/basic.cu`
- `src/gemm.cu`
