#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 08:16:00 2020

@author: au504946
"""



def readDNSdata(inputfilename, onlyU=False):
    from numba import jit
    """
    Function used to read the raw field* files into python
    :param inputfilename: The path to the field file
    :param onlyU: True if only the U parameter is needed
    :return:
        -quantities - the different quantities as the rows
        -quanList - A list over what the rows in quantities are
        -xF - A array of location of the different meshpoints in the x direction
        -yF - A array of location of the different meshpoint in the y direction. Variable stepsize
        -zF - A array of location of the different meshpoint in the z direction
        -length - [0]=z dim, [1]=y dim, [2]=time(s)
        -storl - meshpoints of the domain in x,y,z
        -paraString - String of different nice parameters
    """
    def readFields(field):
        # print('Reading field: ' + str(field))

        rl = np.zeros((NNx, NNz, NNy))
        il = np.zeros((NNx, NNz, NNy))

        for indz in range(NNz):
            for indy in range(NNy):
                np.fromfile(fid, np.int32, count=1)
                vec = np.fromfile(fid, np.float64, count=NNx * 2)
                rl[:, indz, indy] = vec[::2]
                il[:, indz, indy] = vec[1::2]
                np.fromfile(fid, np.int32, count=1)

        return rl, il

    @jit
    def transformComplex(rl, il, ca, sa):
        hr = rl * ca - il * sa
        il = il * ca + rl * sa
        rl = hr

        return rl, il

    def complexReshape(rl, il):
        size = NNx * NNz * NNy
        complexArray = np.reshape(np.vectorize(complex)(np.reshape(rl, size), np.reshape(il, size)), (NNx, NNz, NNy))

        return complexArray

    def fou2phys(fou):
        shape = np.shape(fou)
        Nx = shape[0] * 2
        Nz = shape[1] + 1
        Ny = shape[2]

        if onlyU == True:
            ncomp = 1
        else:
            ncomp = 7
        NNx = Nx
        NNy = int(Ny / ncomp)
        NNz = Nz

        phys = np.zeros((Nx, Nz, Ny))

        for indy in range(Ny):
            tp = fou[:, :, indy]
            tp = np.insert(tp, int(Nz / 2), np.zeros((int(Nx / 2), 1)).T, axis=1)
            tp_placeholder = np.vstack([tp, np.zeros((1, int(Nz)))])

            Am = np.conj(np.flipud(tp[1:int(Nx / 2), 1]))
            Bm = np.conj(np.flipud(np.fliplr(tp[1:int(Nx / 2), 1:int(Nz)])))

            placeholder = np.insert(Bm, 0, Am, axis=1)

            X = np.vstack([tp_placeholder, placeholder])

            x = np.fft.ifft2(X) * Nx * Nz

            phys[:, :, indy] = np.real(x)

        return phys, NNx, NNy, NNz

    # TODO make function wrapper around this

    # inputfilename = 'end.uu'
    scalar = 4

    import numpy as np

    with open(inputfilename, 'rb') as fid:
        # open parameters
        discard = np.fromfile(fid, np.int32, count=1)
        Re = np.fromfile(fid, np.float64, count=1)
        pou = np.fromfile(fid, np.int32, count=1)
        length = np.fromfile(fid, np.float64, count=4)

        if scalar != 0:
            vscal = np.fromfile(fid, np.float64, count=2 * scalar)
        else:
            vscal = 1

        discard = np.fromfile(fid, np.int32, count=2)

        storl = np.fromfile(fid, np.int32, count=4)
        discard = np.fromfile(fid, np.int32, count=2)

        flowtype = np.fromfile(fid, np.int32, count=1)
        dstar = np.fromfile(fid, np.float64, count=1)
        discard = np.fromfile(fid, np.int32, count=1)

        paraString = 'Re: ' + str(Re[0]) + ', Flowtype: ' + str(flowtype[0]) \
                     + ', Nx: ' + str(storl[0]) + ', Ny: ' + str(storl[1]) \
                     + ', Nz: ' + str(storl[2])
        # print(paraString)

        # Define paramerers based on retrieved data
        NNx = int(storl[0] / 2)
        NNy = storl[1]
        NNz = storl[2]
        Re = Re * dstar
        Lx = length[0] / dstar
        Lz = length[1] / dstar
        Ly = 2 / dstar  # In bls, dstar is defined as 2/boxsize
        t = length[3] / dstar
        realpos = np.kron(np.ones((1, NNx)), [1, 0])  # ???? check

        # Reading fields:

        rlu, ilu = readFields('u')
        if onlyU == False:
            rlv, ilv = readFields('v')
            rlw, ilw = readFields('w')
            rls1, ils1 = readFields('s1')
            rls2, ils2 = readFields('s2')
            rls3, ils3 = readFields('s3')
            rls4, ils4 = readFields('s4')

    scale = 1 / dstar
    padx = 0
    padz = 0
    NxF = 2 * NNx
    NzF = NNz
    xF = np.transpose(Lx / NxF * (range(int(-NxF / 2), int(NxF / 2) + 1, 1)))
    zF = np.transpose(Lz / NzF * (range(int(-NzF / 2), int(NzF / 2) + 1, 1)))
    yF = np.transpose(scale * (1 + np.cos(np.pi * (np.linspace(0, 1, (NNy))))))
    xF = -xF[0] + xF

    """
    Shift velocity field in the streamwise direction in order
    to move the fringe to the end of the domain for spatial flows
    """

    kxvec = np.linspace(0, (2 * np.pi / Lx[0] * (NNx - 1)), NNx)
    kzvec = np.linspace(0, (2 * np.pi / Lz[0] * (NNz / 2 - 1)), int(NNz / 2))
    kzvec = np.append(kzvec, (-np.flip(kzvec) - 1))

    xs = Lx / 2.
    zs = 0.

    # TODO Should be in a function maybe lambda for speed?
    cx = np.zeros(NNx)
    sx = np.zeros(NNx)
    ca = np.zeros(NNx)
    sa = np.zeros(NNx)
    import time
    startT = time.time()
    # print('start loops for transform...')
    for i in range(NNx):
        argx = -zs * kxvec[i]
        cx[i] = np.cos(argx)
        sx[i] = np.sin(argx)

    for k in range(NNz):
        argz = -zs * kzvec[k]
        for i in range(NNx):
            ca[i] = cx[i] * np.cos(argz) - sx[i] * np.sin(argz)
            sa[i] = cx[i] * np.sin(argx) - sx[i] * np.cos(argz)

        for i in range(NNx):
            rlu[i, k, :], ilu[i, k, :] = transformComplex(rlu[i, k, :], ilu[i, k, :], ca[i], sa[i])
            if onlyU == False:
                rlv[i, k, :], ilv[i, k, :] = transformComplex(rlv[i, k, :], ilv[i, k, :], ca[i], sa[i])
                rlw[i, k, :], ilw[i, k, :] = transformComplex(rlw[i, k, :], ilw[i, k, :], ca[i], sa[i])
                rls1[i, k, :], ils1[i, k, :] = transformComplex(rls1[i, k, :], ils1[i, k, :], ca[i], sa[i])
                rls2[i, k, :], ils2[i, k, :] = transformComplex(rls2[i, k, :], ils2[i, k, :], ca[i], sa[i])
                rls3[i, k, :], ils3[i, k, :] = transformComplex(rls3[i, k, :], ils3[i, k, :], ca[i], sa[i])
                rls4[i, k, :], ils4[i, k, :] = transformComplex(rls4[i, k, :], ils4[i, k, :], ca[i], sa[i])

                # TODO make loop for scalars?
    # print('Excecution time for transform loops: ' + str(round((time.time() - startT))) + ' Sec')
    # print('reshaping...')

    u = complexReshape(rlu, ilu)
    if onlyU == True:
        vel = np.concatenate((u[:, :int(NNz / 2), :], u[:, (int(NNz / 2) + 1):, :]), axis=1)
        velOut = vel
    if onlyU == False:
        v = complexReshape(rlv, ilv)
        w = complexReshape(rlw, ilw)
        s1 = complexReshape(rls1, ils1)
        s2 = complexReshape(rls2, ils2)
        s3 = complexReshape(rls3, ils3)
        s4 = complexReshape(rls4, ils4)

        vel = np.stack((u, v, w, s1, s2, s3, s4), axis=-1)
        vel = np.concatenate((vel[:, :int(NNz / 2), :], vel[:, (int(NNz / 2) + 1):, :]), axis=1)
        res = vel[:, :, :, 0]
        for i in range(7 - 1):
            res = np.concatenate((res, vel[:, :, :, i + 1]), axis=2)

        velOut = res

    # Convert to physcal space
    phys, NNx, NNy, NNz = fou2phys(velOut)

    # Decouple fields
    u = phys[:, :, int(0 * NNy):int(1 * NNy)]
    u = np.moveaxis(u, [1, 2], [2, 1])
    if onlyU == True:
        quanList = 'u'
        quantities = u
    if onlyU == False:
        v = phys[:, :, int(1 * NNy):int(2 * NNy)]
        w = phys[:, :, int(2 * NNy):int(3 * NNy)]
        s1 = phys[:, :, int(3 * NNy):int(4 * NNy)]
        s2 = phys[:, :, int(4 * NNy):int(5 * NNy)]
        s3 = phys[:, :, int(5 * NNy):int(6 * NNy)]
        s4 = phys[:, :, int(6 * NNy):int(7 * NNy)]

        v = np.moveaxis(v, [1, 2], [2, 1])
        w = np.moveaxis(w, [1, 2], [2, 1])
        s1 = np.moveaxis(s1, [1, 2], [2, 1])
        s2 = np.moveaxis(s2, [1, 2], [2, 1])
        s3 = np.moveaxis(s3, [1, 2], [2, 1])
        s4 = np.moveaxis(s4, [1, 2], [2, 1])

        quanList = ['u', 'v', 'w', 's1', 's2', 's3', 's4']
        quantities = [u, v, w, s1, s2, s3, s4]

    return quantities, quanList, xF, yF, zF, length, storl, paraString