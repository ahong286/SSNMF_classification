/**************************************************************************
 *  COL2IMSTEP Rearrange matrix columns into blocks.
 *
 *  A = COL2IMSTEP(B,[MM NN],[N1 N2]) rearranges the columns of B into
 *  sliding N1-by-N2 blocks producing the matrix A of size MM-by-NN. B is
 *  usually the result of calling IM2COLSTEP(...) with a stepsize of 1, or
 *  using Matlab's IM2COL(..,'sliding'). Overlapping blocks are summed in A.
 *
 *  A = COL2IMSTEP(B,[MM NN],[N1 N2],[S1 S2]) arranges the blocks in A with
 *  a step size of (S1,S2) between them. The first block is at A(1:N1,1:N2),
 *  and the rest are at A((1:N1)+i*S1,(1:N2)+j*S2). Overlapping blocks are
 *  summed in A. Note that B is usually the result of calling
 *  IM2COLSTEP(...) with a stepsize of [S1 S2].
 *
 *  A = IM2COLSTEP(B,[MM NN KK],[N1 N2 N3],[S1 S2 S3]) generates a 3-D
 *  output matrix A. The step size [S1 S2 S3] may be omitted, and defaults
 *  to [1 1 1].
 *
 *************************************************************************/


#include <mex.h>
#include <math.h>
#include <stdlib.h>

#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#endif


#define get_3D_element_idx(m, n, p, i, j, k) ((((k) - 1) * (m) * (n)) + (((j) - 1) * (m)) + ((i) - 1))

#define get_2D_element_idx(m, n, i, j) (((j) - 1) * (m) + (i) - 1)

#define get_3D_2D_element_idx(m, n, p, q, i, j, k, t) (get_2D_element_idx(((m) * (n) * (p)), (q), ((get_3D_element_idx((m), (n), (p), (i), (j), (k))) + 1) , (t)))


/* Input Arguments */

#define	B_IN   (prhs[0])
#define N_IN   (prhs[1])
#define SZ_IN  (prhs[2])
#define S_IN   (prhs[3])

/* #define round(x) ((int)((x)+0.5)) */


/* Output Arguments */

#define	X_OUT  (plhs[0])


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *x, *b, *s;
    mwSize sz[3], stepsize[3], n[3], ndims, mm, nn, pp, blocknum;
    mwIndex i, j, k, ii, jj, kk, xStart, yStart, zStart;
    
    
    /* Check for proper number of arguments */
    
    if (nrhs < 3 || nrhs > 4) {
        mexErrMsgTxt("Invalid number of input arguments.");
    }
    else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    
    /* Check the the input dimensions */
    
    if (!mxIsDouble(B_IN) || mxIsComplex(B_IN) || mxGetNumberOfDimensions(B_IN) > 2) {
        mexErrMsgTxt("B should be a double matrix.");
    }
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
    if (nrhs == 4) {
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
    
    if (nrhs == 4) {
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
    
    
    X_OUT = mxCreateNumericArray(ndims, n, mxDOUBLE_CLASS, mxREAL);
    
    b = mxGetPr(B_IN);
    x = mxGetPr(X_OUT);
    
    
    /* n -- output matrix size (image size) */
    /* sz -- block size */
    /* stepsize -- step size */
    
    mm = (mwSize)(ceil(((double)(n[0] - sz[0])) / ((double)stepsize[0])) + 1 + 0.5);
    nn = (mwSize)(ceil(((double)(n[1] - sz[1])) / ((double)stepsize[1])) + 1 + 0.5);
    pp = (mwSize)(ceil(((double)(n[2] - sz[2])) / ((double)stepsize[2])) + 1 + 0.5);
    
    blocknum = mm * nn * pp;
    for (ii = 1; ii <= mm; ++ii) {
        xStart = min((ii - 1) * stepsize[0] + 1, n[0] - sz[0] + 1);
        for (jj = 1; jj <= nn; ++jj) {
            yStart = min((jj - 1) * stepsize[1] + 1, n[1] - sz[1] + 1);
            for (kk = 1; kk <= pp; ++kk) {
                zStart = min((kk - 1) * stepsize[2] + 1, n[2] - sz[2] + 1);
                for (i = 1; i <= sz[0]; ++i) {
                    for (j = 1; j <= sz[1]; ++j) {
                        for (k = 1; k <= sz[2]; ++k) {
                            x[get_3D_element_idx(n[0], n[1], n[2], xStart + i - 1, yStart + j - 1, zStart + k - 1)] += b[get_3D_2D_element_idx(sz[0], sz[1], sz[2], blocknum, i, j, k, (kk - 1) * mm * nn + (jj - 1) * mm + ii)];
                        }
                    }
                }
            }
        }
    }
}



