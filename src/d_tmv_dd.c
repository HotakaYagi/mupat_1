/**
 * @file d_tmv_dd.c
 * @brief It enable to offload double and double-double transposed matrix-vector multiplication to outer C from MATLAB.
 * @author Hotaka Yagi
 * @date last update 2019 Mar 28
 */
#include "mupat.h"
#include <mex.h>
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    mwSize n;                   /* size of matrix */
    mwSize m;
    mwSize l;
    double *in1;              /* input */
    double *in3;
    double *in4;
    double *out1;              /* output matrix */
    double *out2;              /* output matrix */
    
    /* get the value of the input  */
    in1 =mxGetPr(prhs[0]);
    in3 =mxGetPr(prhs[1]);
    in4 =mxGetPr(prhs[2]);
    int omp_threadNum = (int) *mxGetPr(prhs[3]);
    int avx = (int) *mxGetPr(prhs[4]);
    int fma = (int) *mxGetPr(prhs[5]);
    
    /* get dimensions of the input matrix */
    m = (mwSize)mxGetM(prhs[0]); //GYO
    l = (mwSize)mxGetN(prhs[0]); //RETU
    n = (mwSize)mxGetN(prhs[2]);
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(m,n,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(m,n,mxREAL);
    
    /* get a pointer to the real data in the vaiable matrix */
    out1 = mxGetPr(plhs[0]);
    out2 = mxGetPr(plhs[1]);
    
    /* call the computational routine */
    _d_tmv_dd(out1,out2,in1,in3,in4,m,n,l,omp_threadNum,avx,fma);
}