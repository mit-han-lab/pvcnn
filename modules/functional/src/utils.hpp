#ifndef _UTILS_HPP
#define _UTILS_HPP

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x)                                                    \
  AT_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")

#define CHECK_IS_INT(x)                                                        \
  AT_CHECK(x.scalar_type() == at::ScalarType::Int,                             \
           #x " must be an int tensor")

#define CHECK_IS_FLOAT(x)                                                      \
  AT_CHECK(x.scalar_type() == at::ScalarType::Float,                           \
           #x " must be a float tensor")

#endif
