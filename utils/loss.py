import numpy as np
import torch
from torch.utils.cpp_extension import BuildExtension
import os


def build_emd_loss():
    this_file = os.path.dirname(os.path.realpath(__file__))
    if torch.cuda.is_available():
        with_cuda = True
    else:
        with_cuda = False

    if with_cuda:
        if os.path.exists(os.path.join(this_file, "emd/emd_src/_ext/emd_lib_cu")):
            return
    else:
        if os.path.exists(os.path.join(this_file, "emd/emd_src/_ext/emd_lib")):
            return
    sources = ['emd_src/my_lib.c']
    headers = ['emd_src/my_lib.h']
    defines = []

    if with_cuda:
        print('Including CUDA code.')
        sources += ['emd_src/my_lib_cuda.c']
        headers += ['emd_src/my_lib_cuda.h']
        defines += [('WITH_CUDA', None)]

    extra_objects = ['emd_src/emd_cuda.cu.o']
    extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

    if with_cuda:
        ffi = BuildExtension(
            '_ext.emd_lib',
            headers=headers,
            sources=sources,
            define_macros=defines,
            relative_to=__file__,
            with_cuda=with_cuda,
            extra_objects=extra_objects
        )
    else:
        ffi = BuildExtension(
            '_ext.emd_lib_cu',
            headers=headers,
            sources=sources,
            define_macros=defines,
            relative_to=__file__,
            with_cuda=with_cuda,
            extra_objects=extra_objects
        )

    ffi.build_extensions()


def get_loss(trg, src):
    build_emd_loss()


build_emd_loss()