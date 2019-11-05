#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:31:08 2019

@author: nsde
"""

import torch
from torch.utils.cpp_extension import load

module = load(name = 'example',
              sources = ['ops.cpp',
                         'ops.cu'],
              extra_ldflags=['-lcudart','-lcusolver','-lcusparse'],
              verbose=True,
              with_cuda=True)

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

if __name__ == '__main__':
    n = 4
    Arow = [0, 1, 2, 3, 3, 3, 3]
    Acol = [0, 1, 2, 0, 1, 2, 3]
    Aval = [1.0, 2.0, 3.0, 0.1, 0.1, 0.1, 4.0]
    b = [1.0, 1.0, 1.0, 1.0]
    batchsize = 5

    Arow = torch.tensor(Arow)
    Acol = torch.tensor(Acol)
    Aval = torch.tensor(Aval)[None].repeat(batchsize,1)
    Aval += torch.rand_like(Aval)*1e-3
    b = torch.tensor(b)[None].repeat(batchsize,1)
    b += torch.rand_like(b)*1e-3

    for i in range(batchsize):
        scipy_A = csr_matrix((Aval[i].numpy(), (Arow.numpy(), Acol.numpy())), shape=(n,n))
        scipy_b = b[i].numpy()
        scipy_x = spsolve(scipy_A, scipy_b)
        print(scipy_x)

    print('\n')
    # cusolver requires a bit different format for the indices
    Arow = torch.tensor([1, 2, 3, 4, 8])
    Acol = torch.tensor([1, 2, 3, 1, 2, 3, 4])
    
    res = module.operator(Arow.int().contiguous().cuda(), 
                          Acol.int().contiguous().cuda(), 
                          Aval.double().contiguous().cuda(), 
                          b.double().contiguous().cuda())
    print(res.cpu().numpy().round(7))
