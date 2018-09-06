/**************************************************************************
 *COUNTCOVER Covering of signal samples by blocks
 *
 *  CNT = COUNTCOVER(SZ,BLOCKSIZE,STEPSIZE) assumes a p-dimensional signal
 *  of size SZ=[N1 N2 N3] covered by (possibly overlapping) blocks of
 *  size BLOCKSIZE=[M1 M2 M3]. The blocks start at position (1,1,1)
 *  and are shifted between them by steps of size STEPSIZE=[S1 S2 Sp].
 *  COUNTCOVER returns a matrix the same size as the signal, containing in
 *  each entry the number of blocks covering that sample.
 *
 *  A = COUNTCOVER([MM NN],[N1 N2])
 *
 *  A = COUNTCOVER([MM NN],[N1 N2],[S1 S2])
 *
 *  A = COUNTCOVER([MM NN KK],[N1 N2 N3],[S1 S2 S3])
 *
 *************************************************************************/


#include <mex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#endif

#define get_3D_element_idx(m, n, p, i, j, k) ((((k) - 1) * (m) * (n)) + (((j) - 1) * (m)) + ((i) - 1))


/* Input Arguments */
#define N_IN   (prhs[0])
#define SZ_IN  (prhs[1])
#define S_IN   (prhs[2])

/* Output Arguments */
#define	X_OUT  (plhs[0])


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *x, *s;
    mwSize sz[3], stepsize[3], n[3], ndims, mm, nn, pp, *cnt, total_elem_num;
    mwIndex i, j, k, ii, jj, kk, xStart, yStart, zStart;
    
    
    /* Check for proper number of arguments */
    
    if (nrhs < 2 || nrhs > 3) {
        mexErrMsgTxt("Invalid number of input arguments.");
    }
    else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    
    /* Check the the input dimensions */
    
    if (!mxIsDouble(N_IN) || mxIsComplex(N_IN) || mxGetNumberOfDimensions(N_IN) > 2) {
        mexErrMsgTxt("Invalid output matrix size.");
    }
    ndims = mxGetM(N_IN) * mxGetN(N_IN);
    if (ndims < 2 || ndims > 3) {
        mexErrMsgTxt("Output matrix can only be 2-D or 3-D.");
    }
    if (!mxIsDouble(SZ_IN) || mxIsComplex(SZ_IN) || mxGetNumberOfDimensions(SZ_IN) > 2 || mxGetM(SZ_IN)*mxGetN(SZ_IN) != ndims) {
        mexErrMsgTxt("Invalid block size.");
    }
    if (nrhs == 3) {
        if (!mxIsDouble(S_IN) || mxIsComplex(S_IN) || mxGetNumberOfDimensions(S_IN) > 2 || mxGetM(S_IN)*mxGetN(S_IN) != ndims) {
            mexErrMsgTxt("Invalid step size.");
        }
    }
    
    /* Get parameters */
    
    s = mxGetPr(N_IN);
    if (s[0] < 1 || s[1] < 1 || (ndims == 3 && s[2] < 1)) {
        mexErrMsgTxt("Invalid output matrix size.");
    }
    n[0] = (mwSize)(s[0] + 0.01);
    n[1] = (mwSize)(s[1] + 0.01);
    n[2] = ndims == 3 ? (mwSize)(s[2] + 0.01) : 1;
    
    s = mxGetPr(SZ_IN);
    if (s[0] < 1 || s[1] < 1 || (ndims == 3 && s[2] < 1)) {
        mexErrMsgTxt("Invalid block size.");
    }
    sz[0] = (mwSize)(s[0] + 0.01);
    sz[1] = (mwSize)(s[1] + 0.01);
    sz[2] = ndims == 3 ? (mwSize)(s[2] + 0.01) : 1;
    
    if (nrhs == 3) {
        s = mxGetPr(S_IN);
        if (s[0] < 1 || s[1] < 1 || (ndims == 3 && s[2] < 1)) {
            mexErrMsgTxt("Invalid step size.");
        }
        stepsize[0] = (mwSize)(s[0] + 0.01);
        stepsize[1] = (mwSize)(s[1] + 0.01);
        stepsize[2] = ndims == 3 ? (mwSize)(s[2] + 0.01) : 1;
    }
    else {
        stepsize[0] = stepsize[1] = stepsize[2] = 1;
    }
    
    if (n[0] < sz[0] || n[1] < sz[1] || (ndims == 3 && n[2] < sz[2])) {
        mexErrMsgTxt("Block size too large.");
    }
    
    
    /* n -- output matrix size (image size) */
    /* sz -- block size */
    /* stepsize -- step size */
    
    total_elem_num = n[0] * n[1] * n[2];
    
    cnt = (mwSize *)malloc(total_elem_num * sizeof(mwSize));
    if (cnt == NULL) {
        mexErrMsgTxt("Out of memory.");
    }
    memset(cnt, 0, total_elem_num * sizeof(mwSize));
    
    mm = (mwSize)(ceil(((double)(n[0] - sz[0])) / ((double)stepsize[0])) + 1 + 0.5);
    nn = (mwSize)(ceil(((double)(n[1] - sz[1])) / ((double)stepsize[1])) + 1 + 0.5);
    pp = (mwSize)(ceil(((double)(n[2] - sz[2])) / ((double)stepsize[2])) + 1 + 0.5);
    
    for (ii = 1; ii <= mm; ++ii) {
        xStart = min((ii - 1) * stepsize[0] + 1, n[0] - sz[0] + 1);
        for (jj = 1; jj <= nn; ++jj) {
            yStart = min((jj - 1) * stepsize[1] + 1, n[1] - sz[1] + 1);
            for (kk = 1; kk <= pp; ++kk) {
                zStart = min((kk - 1) * stepsize[2] + 1, n[2] - sz[2] + 1);
                for (i = 0; i < sz[0]; ++i) {
                    for (j = 0; j < sz[1]; ++j) {
                        for (k = 0; k < sz[2]; ++k) {
                            cnt[get_3D_element_idx(n[0], n[1], n[2], xStart + i, yStart + j, zStart + k)]++;
                        }
                    }
                }
            }
        }
    }
    
    X_OUT = mxCreateNumericArray(ndims, n, mxDOUBLE_CLASS, mxREAL);
    x = mxGetPr(X_OUT);
    
    for (i = 0; i < total_elem_num; ++i) {
        x[i] = (double)cnt[i];
    }
    
    free(cnt);
    
}


