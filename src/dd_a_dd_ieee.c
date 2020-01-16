/**
 * @file dd_a_dd_ieee.c
 * @brief It enable to offload double-double and double-double vector or matrix addition to outer C from MATLAB.
 * @author Hotaka Yagi
 * @date last update 2019 Mar 28
 */
#include "mupat.h"
#include <mex.h>
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    double *ahi;              /* input scalar */
    double *alo;
    double *bhi;
    double *blo;
    mwSize n;                   /* size of matrix */
    mwSize m;
    double *chi;              /* output matrix */
    double *clo;
    /* get the value of the input  */
    ahi =mxGetPr(prhs[0]);
    alo =mxGetPr(prhs[1]);
    bhi =mxGetPr(prhs[2]);
    blo =mxGetPr(prhs[3]);
    int omp_threadNum = (int) *mxGetPr(prhs[4]);
    int avx = (int) *mxGetPr(prhs[5]);
    
    /* get dimensions of the input matrix */
    m = (mwSize)mxGetM(prhs[0]); //GYO
    n = (mwSize)mxGetN(prhs[0]); //RETU
    /* get dimensions of the input matrix */
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(m,n,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(m,n,mxREAL);
    /* get a pointer to the real data in the output matrix */
    chi = mxGetPr(plhs[0]);
    clo = mxGetPr(plhs[1]);
    
    /* call the computational routine */
    _dd_a_dd_ieee(chi,clo,ahi,alo,bhi,blo,m,n,omp_threadNum,avx);
}
