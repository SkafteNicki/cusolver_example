#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cusolverSp.h>
#include <cusparse.h>

at::Tensor op_cuda(at::Tensor Arow, at::Tensor Acol, at::Tensor Aval, at::Tensor b, at::Tensor out){
    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = NULL;
    cusolverSpHandle_t cusolverH = NULL;
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;
    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
        
    cusolver_status = cusolverSpCreate(&cusolverH);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusparse_status = cusparseCreateMatDescr(&descrA); 
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE); // base-1

    cusolver_status = cusolverSpCreateCsrqrInfo(&info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    
    const int bs = b.size(0);
    const int m = b.size(1);
    const int nA = Aval.size(1);
    auto aval_pointer = Aval.data_ptr<double>();
    auto arow_pointer = Arow.data_ptr<int>();
    auto acol_pointer = Acol.data_ptr<int>();    
    auto b_pointer = b.data_ptr<double>();
    auto out_pointer = out.data_ptr<double>();

    cusolver_status = cusolverSpXcsrqrAnalysisBatched(
        cusolverH, m, m, nA,
        descrA, arow_pointer, acol_pointer, info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusolver_status = cusolverSpDcsrqrBufferInfoBatched(
        cusolverH, m, m, nA,
        descrA, aval_pointer, arow_pointer, acol_pointer,
        bs, info, &size_internal, &size_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    
    cudaMalloc((void**)&buffer_qr, size_qr);

    cusolver_status = cusolverSpDcsrqrsvBatched(
        cusolverH, m, m, nA,
        descrA, aval_pointer, arow_pointer, acol_pointer,
        b_pointer, out_pointer, bs, info, buffer_qr);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    return out;
}
