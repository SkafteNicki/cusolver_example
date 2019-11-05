#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <cusparse_v2.h>

// Cuda forward declaration
at::Tensor op_cuda(at::Tensor Arow, at::Tensor Acol, at::Tensor Aval, at::Tensor b, at::Tensor out);
                              
// Shortcuts for checking
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Function declaration
at::Tensor op(at::Tensor Arow, 
              at::Tensor Acol,  
              at::Tensor Aval, 
              at::Tensor b){ 
    // Do input checking
    CHECK_INPUT(Arow);
    CHECK_INPUT(Acol);
    CHECK_INPUT(Aval);
    CHECK_INPUT(b);
    // Allocate output tensor
    at::Tensor out = torch::zeros_like(b);
    
    // Run operator
    return op_cuda(Arow, Acol, Aval, b, out);
}

// Binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("operator", &op, "Operator");
}
