/**
 * @file d_a_qd.c
 * @brief It enable to offload double and quad-double vector or matrix addition to outer C from MATLAB.
 * @author Hotaka Yagi
 * @date last update 2019 Mar 28
 */
#include "mupat.h"
#include <mex.h>
/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    mwSize n;                   /* size of matrix */
    mwSize m;
    double *in1;              /* input scalar */
    double *in5;              /* input scalar */
    double *in6;
    double *in7;              /* input scalar */
    double *in8;
    double *out1;              /* output matrix */
    double *out2;              /* output matrix */
    double *out3;              /* output matrix */
    double *out4;              /* output matrix */
    
    /* get the value of the input  */
    in1 =mxGetPr(prhs[0]);
    in5 =mxGetPr(prhs[1]);
    in6 =mxGetPr(prhs[2]);
    in7 =mxGetPr(prhs[3]);
    in8 =mxGetPr(prhs[4]);
    int omp_threadNum = (int) *mxGetPr(prhs[5]);
    int avx = (int) *mxGetPr(prhs[6]);
    
    /* get dimensions of the input matrix */
    m = (mwSize)mxGetM(prhs[0]); //GYO
    n = (mwSize)mxGetN(prhs[0]); //RETU
    /* get dimensions of the input matrix */
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(m,n,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(m,n,mxREAL);
    plhs[2] = mxCreateDoubleMatrix(m,n,mxREAL);
    plhs[3] = mxCreateDoubleMatrix(m,n,mxREAL);
    /* get a pointer to the real data in the output matrix */
    out1 = mxGetPr(plhs[0]);
    out2 = mxGetPr(plhs[1]);
    out3 = mxGetPr(plhs[2]);
    out4 = mxGetPr(plhs[3]);
    
    /* call the computational routine */
    _d_a_qd(out1,out2,out3,out4,in1,in5,in6,in7,in8,m,n,omp_threadNum,avx);  
}