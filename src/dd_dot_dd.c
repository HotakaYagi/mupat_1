/**
 * @file dd_dot_dd.c
 * @brief It enable to offload double-double and double-double inner product to outer C from MATLAB.
 * @author Hotaka Yagi
 * @date last update 2019 Mar 28
 */
#include "mupat.h"
#include <mex.h>
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    double *ahi;              /* input scalar */
    double *alo;
    double *bhi;              /* input scalar */
    double *blo;
    mwSize m;
    mwSize n;
    double *chi;              /* output matrix */
    double *clo;
    /* get the value of the input  */
    ahi =mxGetPr(prhs[0]);
    alo =mxGetPr(prhs[1]);
    bhi =mxGetPr(prhs[2]);
    blo =mxGetPr(prhs[3]);
    int omp_threadNum = (int) *mxGetPr(prhs[4]);
    int avx = (int) *mxGetPr(prhs[5]);
    int fma = (int) *mxGetPr(prhs[6]);
    /* get dimensions of the input matrix */
    m = (mwSize)mxGetM(prhs[0]); //GYO
    n = (mwSize)mxGetN(prhs[0]); //GYO
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
    /* get a pointer to the real data in the output matrix */
    chi = mxGetPr(plhs[0]);
    clo = mxGetPr(plhs[1]);
    /* call the computational routine */
    _dd_dot_dd(chi,clo,ahi,alo,bhi,blo,m,n, omp_threadNum, avx, fma);
}