/**
 * @file dd_scl_d.c
 * @brief It enable to offload double-double and double scaler to outer C from MATLAB.
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
    mwSize k;
    double *in1;              /* input */
    double *in2;
    double *in3;
    double *out1;              /* output matrix */
    double *out2;              /* output matrix */
    
    /* get the value of the input  */
    in1 =mxGetPr(prhs[0]);
    in2 =mxGetPr(prhs[1]);
    in3 =mxGetPr(prhs[2]);
    int omp_threadNum = (int) *mxGetPr(prhs[3]);
    int avx = (int) *mxGetPr(prhs[4]);
    int fma = (int) *mxGetPr(prhs[5]);
    
    /* get dimensions of the input matrix */
    m = (mwSize)mxGetM(prhs[0]); //GYO
    n = (mwSize)mxGetN(prhs[0]); //RETU
    l = (mwSize)mxGetM(prhs[2]);
    k = (mwSize)mxGetN(prhs[2]);
    
    if ((l==1)&&(k==1)){ //ddx-da
        /* create the output matrix */
        plhs[0] = mxCreateDoubleMatrix(m,n,mxREAL);
        plhs[1] = mxCreateDoubleMatrix(m,n,mxREAL);
        
        /* get a pointer to the real data in the vaiable matrix */
        out1 = mxGetPr(plhs[0]);
        out2 = mxGetPr(plhs[1]);
        
        /* call the computational routine */
        _d_scl_dd(out1,out2,in3,in1,in2,m,n,omp_threadNum,avx,fma);
    }else if ((m==1)&&(n==1)){ //dda-dx
        
        /* create the output matrix */
        plhs[0] = mxCreateDoubleMatrix(l,k,mxREAL);
        plhs[1] = mxCreateDoubleMatrix(l,k,mxREAL);
        
        /* get a pointer to the real data in the vaiable matrix */
        out1 = mxGetPr(plhs[0]);
        out2 = mxGetPr(plhs[1]);
        
        /* call the computational routine */
        _dd_scl_d(out1,out2,in1,in2,in3,l,k,omp_threadNum,avx,fma);
    }
}