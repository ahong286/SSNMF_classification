/**************************************************************************
 *IM2COLSTEP Rearrange matrix blocks into columns.
 *
 *  B = IM2COLSTEP(A,[N1 N2]) converts each sliding N1-by-N2 block of the
 *  2-D matrix A into a column of B, with no zero padding. B has N1*N2 rows
 *  and will contain as many columns as there are N1-by-N2 neighborhoods in
 *  A. Each column of B contains a neighborhood of A reshaped as NHOOD(:),
 *  where NHOOD is a matrix containing an N1-by-N2 neighborhood of A.
 *
 *  B = IM2COLSTEP(A,[N1 N2],[S1 S2]) extracts neighborhoods of A with a
 *  step size of (S1,S2) between them. The first extracted neighborhood is
 *  A(1:N1,1:N2), and the rest are of the form A((1:N1)+i*S1,(1:N2)+j*S2).
 *  Note that to ensure coverage of all A by neighborhoods,
 *  (size(A,i)-Ni)/Si must be whole for i=1,2. The default function behavior
 *  corresponds to [S1 S2] = [1 1]. Setting S1>=N1 and S2>=N2 results in no
 *  overlap between the neighborhoods.
 *
 *  B = IM2COLSTEP(A,[N1 N2 N3],[S1 S2 S3]) operates on a 3-D matrix A. The
 *  step size [S1 S2 S3] may be omitted, and defaults to [1 1 1].
 *
 *************************************************************************/


#include <mex.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#endif


#define get_3D_element_idx(m, n, p, i, j, k) ((((k) - 1) * (m) * (n)) + (((j) - 1) * (m)) + ((i) - 1))

#define get_2D_element_idx(m, n, i, j) (((j) - 1) * (m) + (i) - 1)

#define get_3D_2D_element_idx(m, n, p, q, i, j, k, t) (get_2D_element_idx(((m) * (n) * (p)), (q), ((get_3D_element_idx((m), (n), (p), (i), (j), (k))) + 1) , (t)))


/* Input Arguments */

#define X_IN  (prhs[0])
#define SZ_IN (prhs[1])
#define S_IN  (prhs[2])


/* Output Arguments */
#define B_OUT (plhs[0])
#define B_POS (plhs[1])

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *x, *b, *s, *b_pos;
    mwSize sz[3], stepsize[3], n[3], ndims, mm, nn, pp, blocknum;
    mwIndex i, j, k, ii, jj, kk, xStart, yStart, zStart;
    
    
    /* Check for proper number of arguments */
    
    if (nrhs < 2 || nrhs > 3) {
        mexErrMsgTxt("Invalid number of input arguments.");
    }
    else if (nlhs > 2) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    
    /* Check the the input dimensions */
    
    ndims = mxGetNumberOfDimensions(X_IN);
    
    if (!mxIsDouble(X_IN) || mxIsComplex(X_IN) || ndims > 3) {
        mexErrMsgTxt("X should be a 2-D or 3-D double matrix.");
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
    
    n[0] = (mxGetDimensions(X_IN))[0];
    n[1] = (mxGetDimensions(X_IN))[1];
    n[2] = ndims == 3 ? (mxGetDimensions(X_IN))[2] : 1;
    
    if (n[0] < sz[0] || n[1] < sz[1] || (ndims == 3 && n[2] < sz[2])) {
        mexErrMsgTxt("Block size too large.");
    }
    
    
    
    /* n -- output matrix size (image size) */
    /* sz -- block size */
    /* stepsize -- step size */
    
    mm = (mwSize)(ceil(((double)(n[0] - sz[0])) / ((double)stepsize[0])) + 1 + 0.5);
    nn = (mwSize)(ceil(((double)(n[1] - sz[1])) / ((double)stepsize[1])) + 1 + 0.5);
    pp = (mwSize)(ceil(((double)(n[2] - sz[2])) / ((double)stepsize[2])) + 1 + 0.5);
    
    blocknum = mm * nn * pp;
        
    /* Create a matrix for the return argument */
    
    B_OUT = mxCreateDoubleMatrix(sz[0] * sz[1] * sz[2], blocknum, mxREAL);
    B_POS = mxCreateDoubleMatrix( blocknum, 2, mxREAL);
    
    /* Assign pointers */
    
    x = mxGetPr(X_IN);
    b = mxGetPr(B_OUT);
    b_pos = mxGetPr(B_POS); 
    
    #pragma omp parallel for private(ii, jj, kk, i, j, k, xStart, yStart, zStart) schedule(dynamic)
    for (ii = 1; ii <= mm; ++ii) {
        xStart = min((ii - 1) * stepsize[0] + 1, n[0] - sz[0] + 1);
        for (jj = 1; jj <= nn; ++jj) {
            yStart = min((jj - 1) * stepsize[1] + 1, n[1] - sz[1] + 1);
            for (kk = 1; kk <= pp; ++kk) {
                zStart = min((kk - 1) * stepsize[2] + 1, n[2] - sz[2] + 1);
                b_pos[(ii - 1) * nn + jj-1] = yStart + sz[1]/2;
                b_pos[blocknum + (ii - 1) * nn + jj-1] = xStart + sz[0]/2 ;
                for (i = 1; i <= sz[0]; ++i) {
                    for (j = 1; j <= sz[1]; ++j) {
                        for (k = 1; k <= sz[2]; ++k) {
                            b[get_3D_2D_element_idx(sz[0], sz[1], sz[2], blocknum, i, j, k, (kk - 1) * mm * nn + (jj - 1) * mm + ii)] = x[get_3D_element_idx(n[0], n[1], n[2], xStart + i - 1, yStart + j - 1, zStart + k - 1)];
                        }
                    }
                }
            }
        }
    }
}



