/**
 * @file d_a_dd.c
 * @brief It enable to offload double and double-double vector or matrix addition to outer C from MATLAB.
 * @author Hotaka Yagi
 * @date last update 2019 Mar 28
 */

#include "mupat.h"
#include <mex.h>
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]){
    /* input matrix */
    double *ahi;
    double *bhi;
    double *blo;
    /* size of matrix */
    mwSize n;
    mwSize m;
    /* output matrix */
    double *chi;
    double *clo;
    /* get the value of the input  */
    ahi =mxGetPr(prhs[0]);
    bhi =mxGetPr(prhs[1]);
    blo =mxGetPr(prhs[2]);
    int omp_threadNum = (int) *mxGetPr(prhs[3]);
    int avx = (int) *mxGetPr(prhs[4]);
    /* get dimensions of the input matrix */
    m = (mwSize)mxGetM(prhs[0]); 
    n = (mwSize)mxGetN(prhs[0]); 
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(m,n,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(m,n,mxREAL);
    /* get a pointer to the real data in the output matrix */
    chi = mxGetPr(plhs[0]);
    clo = mxGetPr(plhs[1]);
    /* call the computational routine */
    _d_a_dd(chi,clo,ahi,bhi,blo,m,n,omp_threadNum,avx);
}