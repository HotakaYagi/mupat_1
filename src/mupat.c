/**
 * @file mupat.c
 * @brief functions in MuPAT
 * @author Hotaka Yagi
 * @date last update 2020 Jan 15
 */

#include "mupat.h"

//-----------------------------------------------------------------
// Double - Double-Double operation
//-----------------------------------------------------------------

/**
 *
 * @fn _d_a_dd(double*,double*,double*,double*,double*,int,int,int,int)
 * @brief You can compute the addition of double and do uble-double numbers in a vector or matrix. You can choose the number of OpenMP threads and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _d_a_dd(double *chi, double *clo, double *ahi, double *bhi, double *blo, int m, int n, int omp_threadNum, int avx)
{
    int i;
    int p = m % 4;
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    switch(avx){
        case 0:
#pragma omp parallel num_threads(omp_threadNum)
        {
            double sahi,sbhi,sblo,schi,sclo;
#pragma omp for
            for (i = 0; i < m * n; i ++){
                //load
                sahi = ahi[i];
                sbhi = bhi[i];
                sblo = blo[i];
                
                //calc
                _d_add_dd(&schi,&sclo,sahi,sbhi,sblo);
                
                //store
                chi[i] = schi;
                clo[i] = sclo;
            }
        }
        break;
        case 1:
#pragma omp parallel num_threads(omp_threadNum)
        {
            double sahi,sbhi,sblo,schi,sclo;
            __m256d vahi,vbhi,vblo,vchi,vclo;
#pragma omp for
            for (i = 0; i < m * n - p; i += 4){
                //load
                vahi = _mm256_loadu_pd(&ahi[i]);
                vbhi = _mm256_loadu_pd(&bhi[i]);
                vblo = _mm256_loadu_pd(&blo[i]);
                
                //calc
                _d_add_dd_avx2(&vchi,&vclo,vahi,vbhi,vblo);
                
                //store
                _mm256_storeu_pd( &chi[i], vchi );
                _mm256_storeu_pd( &clo[i], vclo );
            }
#pragma omp for
            for (i = m * n - p; i < m * n; i ++){
                //load
                sahi = ahi[i];
                sbhi = bhi[i];
                sblo = blo[i];
                
                //calc
                _d_add_dd(&schi,&sclo,sahi,sbhi,sblo);
                
                //store
                chi[i] = schi;
                clo[i] = sclo;
            }
        }
        break;
        
    }
}
/**
 *
 * @fn _d_scl_dd(double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the scaler of double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _d_scl_dd(double *chi, double *clo, double *ahi, double *bhi, double *blo,int m, int n, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,sbhi,sblo,schi,sclo;
                    sahi = ahi[0];
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                        
                        //store
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahi,valo,vbhi,vblo,vchi,vclo,vch,vcl;
                    double vzhi[4]={0,0,0,0};
                    double vzlo[4]={0,0,0,0};
                    vahi = _mm256_set_pd(ahi[0],0,0,0);
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vbhi = _mm256_set_pd(bhi[i],0,0,0);
                        vblo = _mm256_set_pd(blo[i],0,0,0);
                        
                        //calc
                        _d_mul_dd_fma(&vchi,&vclo,vahi,vbhi,vblo);
                        
                        _mm256_storeu_pd( &vzhi[0], vchi );
                        _mm256_storeu_pd( &vzlo[0], vclo );
                        
                        chi[i]=vzhi[3];
                        clo[i]=vzlo[3];
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,sbhi,sblo,schi,sclo;
                    
                    __m256d vahi,valo,vbhi,vblo,vchi,vclo,vch,vcl;
                    sahi = ahi[0];
                    vahi = _mm256_broadcast_sd(&ahi[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhi = _mm256_loadu_pd(&bhi[i]);
                        vblo = _mm256_loadu_pd(&blo[i]);
                        
                        //calc
                        _d_mul_dd_avx2(&vchi,&vclo,vahi,vbhi,vblo);
                        
                        //store
                        _mm256_storeu_pd( &chi[i], vchi );
                        _mm256_storeu_pd( &clo[i], vclo );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                        
                        //store
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,sbhi,sblo,schi,sclo;
                    
                    __m256d vahi,valo,vbhi,vblo,vchi,vclo,vch,vcl;
                    sahi = ahi[0];
                    vahi = _mm256_broadcast_sd(&ahi[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhi = _mm256_loadu_pd(&bhi[i]);
                        vblo = _mm256_loadu_pd(&blo[i]);
                        
                        //calc
                        _d_mul_dd_fma(&vchi,&vclo,vahi,vbhi,vblo);
                        
                        //store
                        _mm256_storeu_pd( &chi[i], vchi );
                        _mm256_storeu_pd( &clo[i], vclo );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                        
                        //store
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
            }
    }
}
/**
 *
 * @fn _d_dot_dd(double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the inner product of double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _d_dot_dd(double *chi, double *clo, double *ahi, double *bhi, double *blo,int m, int n, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahi,sbhi,sblo,schi,sclo,szhi,szlo;
                    double ch [num_threads];
                    double cl [num_threads];
                    for (i = 0; i < num_threads; i ++){
                        ch[i]=0;
                        cl[i]=0;
                    }
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sahi = ahi[i];
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                        _dd_add_dd(&szhi,&szlo,ch[tid],cl[tid],schi,sclo);
                        
                        //store
                        ch[tid] = szhi;
                        cl[tid] = szlo;
                    }
#pragma omp critical
                    {
                        //calc
                        _dd_add_dd(&schi,&sclo,ch[tid],cl[tid],chi[0],clo[0]);
                        
                        //store
                        chi[0] = schi;
                        clo[0] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double schi,sclo;
                    double ch [num_threads*4];
                    double cl [num_threads*4];
                    for (i = 0; i < num_threads * 4; i ++){
                        ch[i]=0;
                        cl[i]=0;
                    }
                    __m256d vahi,vbhi,vblo,vzhi,vzlo;
                    __m256d vchi = _mm256_setzero_pd();
                    __m256d vclo = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vahi = _mm256_set_pd(ahi[i],0,0,0);
                        vbhi = _mm256_set_pd(bhi[i],0,0,0);
                        vblo = _mm256_set_pd(blo[i],0,0,0);
                        
                        //calc
                        _d_mul_dd_fma(&vzhi,&vzlo,vahi,vbhi,vblo);
                        _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                        
                    }
                    //store
                    _mm256_storeu_pd( &ch[4*tid], vchi );
                    _mm256_storeu_pd( &cl[4*tid], vclo );
                    
#pragma omp critical
                    {
                        //calc
                        _dd_add_dd(&schi,&sclo,ch[4*tid+3],cl[4*tid+3],chi[0],clo[0]);
                        
                        //store
                        chi[0] = schi;
                        clo[0] = sclo;
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahi,sbhi,sblo,schi,sclo,szhi,szlo;
                    double ch [num_threads*4];
                    double cl [num_threads*4];
                    for (i = 0; i < num_threads * 4; i ++){
                        ch[i]=0;
                        cl[i]=0;
                    }
                    __m256d vahi,vbhi,vblo,vzhi,vzlo;
                    __m256d vchi = _mm256_setzero_pd();
                    __m256d vclo = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahi = _mm256_loadu_pd(&ahi[i]);
                        vbhi = _mm256_loadu_pd(&bhi[i]);
                        vblo = _mm256_loadu_pd(&blo[i]);
                        
                        //calc
                        _d_mul_dd_avx2(&vzhi,&vzlo,vahi,vbhi,vblo);
                        _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                        
                    }
                    //store
                    _mm256_storeu_pd( &ch[4*tid], vchi );
                    _mm256_storeu_pd( &cl[4*tid], vclo );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        //load
                        sahi = ahi[i];
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _d_mul_dd(&szhi,&szlo,sahi,sbhi,sblo);
                        _dd_add_dd(&schi,&sclo,szhi,szlo,ch[4*tid],cl[4*tid]);
                        
                        //store
                        ch[4*tid] = schi;
                        cl[4*tid] = sclo;
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _dd_add_dd(&schi,&sclo,ch[4*tid+i],cl[4*tid+i],ch[4*tid],cl[4*tid]);
                        
                        //store
                        ch[4*tid] = schi;
                        cl[4*tid] = sclo;
                    }
#pragma omp critical
                    {
                        //calc
                        _dd_add_dd(&schi,&sclo,ch[4*tid],cl[4*tid],chi[0],clo[0]);
                        
                        //store
                        chi[0] = schi;
                        clo[0] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahi,sbhi,sblo,schi,sclo,szhi,szlo;
                    double ch [num_threads*4];
                    double cl [num_threads*4];
                    for (i = 0; i < num_threads * 4; i ++){
                        ch[i]=0;
                        cl[i]=0;
                    }
                    __m256d vahi,vbhi,vblo,vzhi,vzlo;
                    __m256d vchi = _mm256_setzero_pd();
                    __m256d vclo = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahi = _mm256_loadu_pd(&ahi[i]);
                        vbhi = _mm256_loadu_pd(&bhi[i]);
                        vblo = _mm256_loadu_pd(&blo[i]);
                        
                        //calc
                        _d_mul_dd_fma(&vzhi,&vzlo,vahi,vbhi,vblo);
                        _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                        
                    }
                    //store
                    _mm256_storeu_pd( &ch[4*tid], vchi );
                    _mm256_storeu_pd( &cl[4*tid], vclo );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        //load
                        sahi = ahi[i];
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _d_mul_dd(&szhi,&szlo,sahi,sbhi,sblo);
                        _dd_add_dd(&schi,&sclo,szhi,szlo,ch[4*tid],cl[4*tid]);
                        
                        //store
                        ch[4*tid] = schi;
                        cl[4*tid] = sclo;
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _dd_add_dd(&schi,&sclo,ch[4*tid+i],cl[4*tid+i],ch[4*tid],cl[4*tid]);
                        
                        //store
                        ch[4*tid] = schi;
                        cl[4*tid] = sclo;
                    }
#pragma omp critical
                    {
                        //calc
                        _dd_add_dd(&schi,&sclo,ch[4*tid],cl[4*tid],chi[0],clo[0]);
                        
                        //store
                        chi[0] = schi;
                        clo[0] = sclo;
                    }
                }
                break;
            }
    }
}
/**
 *
 * @fn _d_mv_dd(double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-vector multiplication of double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _d_mv_dd(double *chi, double *clo, double *ahi,double *bhi, double *blo,int m, int n, int l, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int k;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        ch[k]=0;
                        cl[k]=0;
                    }
                    double sahi,sbhi,sblo,schi,sclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k ++) {
                        //load
                        sbhi = bhi[k];
                        sblo = blo[k];
                        for (i = 0; i < m; i ++) {
                            //load
                            sahi = ahi[m*k+i];
                            
                            //calc
                            _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i = 0; i < m; i ++){
                        
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double zhi[4] = {0,0,0,0};
                    double zlo[4] = {0,0,0,0};
                    double ch [num_threads * m];
                    double cl [num_threads * m];
                    for (k = 0; k < num_threads * m; k ++) {
                        ch[k] = 0;
                        cl[k] = 0;
                    }
                    double sahi,sbhi,sblo,schi,sclo;
                    __m256d vahi,vbhi,vblo,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        //load
                        vbhi = _mm256_set_pd(bhi[k],0,0,0);
                        vblo = _mm256_set_pd(blo[k],0,0,0);
                        
                        for (i = 0; i < m; i++) {
                            //load
                            vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                            vchi = _mm256_set_pd(ch[m*tid+i],0,0,0);
                            vclo = _mm256_set_pd(cl[m*tid+i],0,0,0);
                            
                            //calc
                            _d_mul_dd_fma(&vzhi,&vzlo,vahi,vbhi,vblo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &zhi[0], vchi );
                            _mm256_storeu_pd( &zlo[0], vclo );
                            
                            ch[m*tid+i]=zhi[3];
                            cl[m*tid+i]=zlo[3];
                        }
                    }
#pragma omp critical
                    for(i = 0; i < m; i ++){
                        
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k, i, j;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    for (k = 0; k < num_threads * m; k++) {
                        ch[k] = 0;
                        cl[k] = 0;
                    }
                    double sahi,sbhi,sblo,schi,sclo;
                    __m256d vahi,vbhi,vblo,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhi = _mm256_broadcast_sd(&bhi[k]);
                        vblo = _mm256_broadcast_sd(&blo[k]);
                        
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            vchi = _mm256_loadu_pd(&ch[m*tid+i]);
                            vclo = _mm256_loadu_pd(&cl[m*tid+i]);
                            
                            //calc
                            _d_mul_dd_avx2(&vzhi,&vzlo,vahi,vbhi,vblo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &ch[m*tid+i], vchi );
                            _mm256_storeu_pd( &cl[m*tid+i], vclo );
                            
                        }
                        sbhi = bhi[k];
                        sblo = blo[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            
                            //calc
                            _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k, i, j;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        ch[k]=0;
                        cl[k]=0;
                    }
                    double sahi,sbhi,sblo,schi,sclo;
                    __m256d vahi,vbhi,vblo,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhi = _mm256_broadcast_sd(&bhi[k]);
                        vblo = _mm256_broadcast_sd(&blo[k]);
                        
                        for (i = 0; i < m-p; i+=4) {
                            
                            //load
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            vchi = _mm256_loadu_pd(&ch[m*tid+i]);
                            vclo = _mm256_loadu_pd(&cl[m*tid+i]);
                            
                            //calc
                            _d_mul_dd_fma(&vzhi,&vzlo,vahi,vbhi,vblo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &ch[m*tid+i], vchi );
                            _mm256_storeu_pd( &cl[m*tid+i], vclo );
                            
                        }
                        sbhi = bhi[k];
                        sblo = blo[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            
                            //calc
                            _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
            }
            break;
    }
}
/**
 *
 * @fn _d_tmv_dd(double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the transposed matrix-vector multiplication of double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _d_tmv_dd( double *chi, double *clo, double *ahi,double *bhi, double *blo,int m, int n, int l, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int k;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i;
                    double sahi,sbhi,sblo,schi,sclo,ch,cl;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        ch=0;
                        cl=0;
                        for (i = 0; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            sbhi = bhi[i];
                            sblo = blo[i];
                            //calc
                            _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                            _dd_add_dd(&ch,&cl,schi,sclo,ch,cl);
                            
                        }
                        chi[k]=ch;
                        clo[k]=cl;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int k,i;
                    __m256d vahi,vbhi,vblo,vchi,vclo,vzhi,vzlo;
                    double zhi[4]={0,0,0,0};
                    double zlo[4]={0,0,0,0};
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vchi = _mm256_set_pd(0,0,0,0);
                        vclo = _mm256_set_pd(0,0,0,0);
                        
                        for (i = 0; i < m; i++) {
                            //load
                            vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                            vbhi = _mm256_set_pd(bhi[i],0,0,0);
                            vblo = _mm256_set_pd(blo[i],0,0,0);
                            
                            //calc
                            _d_mul_dd_fma(&vzhi,&vzlo,vahi,vbhi,vblo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                        }
                        _mm256_storeu_pd( &zhi[0], vchi );
                        _mm256_storeu_pd( &zlo[0], vclo );
                        
                        chi[k]=zhi[3];
                        clo[k]=zlo[3];
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k;
                    double ch[4]={0,0,0,0};
                    double cl[4]={0,0,0,0};
                    double sahi,sbhi,sblo,schi,sclo;
                    __m256d vahi,vbhi,vblo,vzhi,vzlo,vchi,vclo,vch,vcl;
                    int i,j;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vch = _mm256_set_pd(0,0,0,0);
                        vcl = _mm256_set_pd(0,0,0,0);
                        
                        for (i = 0; i < m-p; i+=4) {
                            
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            vbhi = _mm256_loadu_pd(&bhi[i]);
                            vblo = _mm256_loadu_pd(&blo[i]);
                            
                            //calc
                            _d_mul_dd_avx2(&vzhi,&vzlo,vahi,vbhi,vblo);
                            _dd_add_dd_avx2(&vch,&vcl,vzhi,vzlo,vch,vcl);
                        }
                        _mm256_storeu_pd( &ch[0], vch );
                        _mm256_storeu_pd( &cl[0], vcl );
                        
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            sbhi = bhi[i];
                            sblo = blo[i];
                            
                            //calc
                            _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[0],cl[0]);
                            
                            
                            //store
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        
                        for(i=1;i<4;i++){
                            
                            _dd_add_dd(&schi,&sclo,ch[0],cl[0],ch[i],cl[i]);
                            
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        chi[k]=ch[0];
                        clo[k]=cl[0];
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k;
                    double ch[4]={0,0,0,0};
                    double cl[4]={0,0,0,0};
                    double sahi,sbhi,sblo,schi,sclo;
                    __m256d vahi,vbhi,vblo,vzhi,vzlo,vchi,vclo,vch,vcl;
                    int i,j;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vch = _mm256_set_pd(0,0,0,0);
                        vcl = _mm256_set_pd(0,0,0,0);
                        
                        for (i = 0; i < m-p; i+=4) {
                            
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            vbhi = _mm256_loadu_pd(&bhi[i]);
                            vblo = _mm256_loadu_pd(&blo[i]);
                            
                            //calc
                            _d_mul_dd_fma(&vzhi,&vzlo,vahi,vbhi,vblo);
                            _dd_add_dd_avx2(&vch,&vcl,vzhi,vzlo,vch,vcl);
                        }
                        _mm256_storeu_pd( &ch[0], vch );
                        _mm256_storeu_pd( &cl[0], vcl );
                        
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            sbhi = bhi[i];
                            sblo = blo[i];
                            
                            //calc
                            _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[0],cl[0]);
                            
                            
                            //store
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        
                        for(i=1;i<4;i++){
                            
                            _dd_add_dd(&schi,&sclo,ch[0],cl[0],ch[i],cl[i]);
                            
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        chi[k]=ch[0];
                        clo[k]=cl[0];
                    }
                }
                break;
            }
            break;
    }
}
/**
 *
 * @fn _d_mm_dd(double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-matrix multiplication of double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _d_mm_dd(double *chi, double *clo, double *ahi,double *bhi, double *blo,int m, int n, int l, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int k;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,sbhi,sblo,schi,sclo;
                    int i,j,k;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            for (i = 0; i < m; i++) {
                                //load
                                sahi = ahi[m*k+i];
                                sbhi = bhi[l*j+k];
                                sblo = blo[l*j+k];
                                
                                //calc
                                _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                                _dd_add_dd(&schi,&sclo,schi,sclo,chi[m*j+i],clo[m*j+i]);
                                
                                //store
                                chi[m*j+i] = schi;
                                clo[m*j+i] = sclo;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahi,vbhi,vblo,vchi,vclo,vzhi,vzlo;
                    double zhi[4]={0,0,0,0};
                    double zlo[4]={0,0,0,0};
                    int j,k,i;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhi = _mm256_set_pd(bhi[l*j+k],0,0,0);
                            vblo = _mm256_set_pd(blo[l*j+k],0,0,0);
                            for (i = 0; i < m; i++) {
                                vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                                vchi = _mm256_set_pd(chi[m*j+i],0,0,0);
                                vclo = _mm256_set_pd(clo[m*j+i],0,0,0);
                                
                                //calc
                                _d_mul_dd_fma(&vzhi,&vzlo,vahi,vbhi,vblo);
                                _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                                
                                _mm256_storeu_pd( &zhi[0], vchi );
                                _mm256_storeu_pd( &zlo[0], vclo );
                                
                                chi[m*j+i]=zhi[3];
                                clo[m*j+i]=zlo[3];
                            }
                        }
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    int p=m%4;
                    double sahi,sbhi,sblo,schi,sclo;
                    __m256d vahi,vbhi,vblo,vzhi,vzlo,vchi,vclo,vch,vcl;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhi = _mm256_broadcast_sd(&bhi[l*j+k]);
                            vblo = _mm256_broadcast_sd(&blo[l*j+k]);
                            for (i = 0; i < m-p; i+=4) {
                                vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                                vchi = _mm256_loadu_pd(&chi[m*j+i]);
                                vclo = _mm256_loadu_pd(&clo[m*j+i]);
                                
                                //calc
                                _d_mul_dd_avx2(&vzhi,&vzlo,vahi,vbhi,vblo);
                                _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                                
                                _mm256_storeu_pd( &chi[m*j+i], vchi );
                                _mm256_storeu_pd( &clo[m*j+i], vclo );
                            }
                            for (i = m-p; i < m; i++) {
                                //load
                                sahi = ahi[m*k+i];
                                sbhi = bhi[l*j+k];
                                sblo = blo[l*j+k];
                                
                                //calc
                                _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                                _dd_add_dd(&schi,&sclo,schi,sclo,chi[m*j+i],clo[m*j+i]);
                                
                                //store
                                chi[m*j+i] = schi;
                                clo[m*j+i] = sclo;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    int p=m%4;
                    double sahi,sbhi,sblo,schi,sclo;
                    __m256d vahi,vbhi,vblo,vzhi,vzlo,vchi,vclo,vch,vcl;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhi = _mm256_broadcast_sd(&bhi[l*j+k]);
                            vblo = _mm256_broadcast_sd(&blo[l*j+k]);
                            for (i = 0; i < m-p; i+=4) {
                                vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                                vchi = _mm256_loadu_pd(&chi[m*j+i]);
                                vclo = _mm256_loadu_pd(&clo[m*j+i]);
                                
                                //calc
                                _d_mul_dd_fma(&vzhi,&vzlo,vahi,vbhi,vblo);
                                _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                                
                                _mm256_storeu_pd( &chi[m*j+i], vchi );
                                _mm256_storeu_pd( &clo[m*j+i], vclo );
                            }
                            for (i = m-p; i < m; i++) {
                                //load
                                sahi = ahi[m*k+i];
                                sbhi = bhi[l*j+k];
                                sblo = blo[l*j+k];
                                
                                //calc
                                _d_mul_dd(&schi,&sclo,sahi,sbhi,sblo);
                                _dd_add_dd(&schi,&sclo,schi,sclo,chi[m*j+i],clo[m*j+i]);
                                
                                //store
                                chi[m*j+i] = schi;
                                clo[m*j+i] = sclo;
                            }
                        }
                    }
                }
                break;
            }
            break;
    }
}


//-----------------------------------------------------------------
// Double - Quad-Double operation
//-----------------------------------------------------------------

/**
 *
 * @fn _d_a_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int)
 * @brief You can compute the addition of double and quad-double numbers in a vector or matrix. You can choose the number of OpenMP threads and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh input (double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _d_a_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int omp_threadNum, int avx)
{
    int i;
    int p = m % 4;
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    switch(avx){
        case 0:
#pragma omp parallel num_threads(omp_threadNum)
        {
            double sahh,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
#pragma omp for
            for (i = 0; i < m * n; i ++){
                //load
                sahh = ahh[i];
                sbhh = bhh[i];
                sbhl = bhl[i];
                sblh = blh[i];
                sbll = bll[i];
                
                //calc
                _d_add_qd(&schh,&schl,&sclh,&scll,sahh,sbhh,sbhl,sblh,sbll);
                
                //store
                chh[i] = schh;
                chl[i] = schl;
                clh[i] = sclh;
                cll[i] = scll;
            }
        }
        break;
        case 1:
#pragma omp parallel num_threads(omp_threadNum)
        {
            double sahh,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
            __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll;
#pragma omp for
            for (i = 0; i < m * n - p; i += 4){
                //load
                vahh = _mm256_loadu_pd(&ahh[i]);
                vbhh = _mm256_loadu_pd(&bhh[i]);
                vbhl = _mm256_loadu_pd(&bhl[i]);
                vblh = _mm256_loadu_pd(&blh[i]);
                vbll = _mm256_loadu_pd(&bll[i]);
                
                //calc
                _d_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vahh,vbhh,vbhl,vblh,vbll);
                
                //store
                _mm256_storeu_pd( &chh[i], vchh );
                _mm256_storeu_pd( &chl[i], vchl );
                _mm256_storeu_pd( &clh[i], vclh );
                _mm256_storeu_pd( &cll[i], vcll );
            }
#pragma omp for
            for (i = m * n - p; i < m * n; i ++){
                //load
                sahh = ahh[i];
                sbhh = bhh[i];
                sbhl = bhl[i];
                sblh = blh[i];
                sbll = bll[i];
                
                //calc
                _d_add_qd(&schh,&schl,&sclh,&scll,sahh,sbhh,sbhl,sblh,sbll);
                
                //store
                chh[i] = schh;
                chl[i] = schl;
                clh[i] = sclh;
                cll[i] = scll;
            }
        }
        break;
        
    }
}

/**
 *
 * @fn _d_scl_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the scaler of double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh input (double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _d_scl_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
                    sahh = ahh[0];
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _d_mul_qd(&schh,&schl,&sclh,&scll,sahh,sbhh,sbhl,sblh,sbll);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll;
                    double vzhh[4]={0,0,0,0};
                    double vzhl[4]={0,0,0,0};
                    double vzlh[4]={0,0,0,0};
                    double vzll[4]={0,0,0,0};
                    vahh = _mm256_set_pd(ahh[0],0,0,0);
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vbhh = _mm256_set_pd(bhh[i],0,0,0);
                        vbhl = _mm256_set_pd(bhl[i],0,0,0);
                        vblh = _mm256_set_pd(blh[i],0,0,0);
                        vbll = _mm256_set_pd(bll[i],0,0,0);
                        
                        //calc
                        _d_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vahh,vbhh,vbhl,vblh,vbll);
                        
                        //store
                        _mm256_storeu_pd( &vzhh[0], vchh );
                        _mm256_storeu_pd( &vzhl[0], vchl );
                        _mm256_storeu_pd( &vzlh[0], vclh );
                        _mm256_storeu_pd( &vzll[0], vcll );
                        
                        chh[i] = vzhh[3];
                        chl[i] = vzhl[3];
                        clh[i] = vzlh[3];
                        cll[i] = vzll[3];
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll;
                    sahh = ahh[0];
                    vahh = _mm256_broadcast_sd(&ahh[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _d_mul_qd_avx2(&vchh,&vchl,&vclh,&vcll,vahh,vbhh,vbhl,vblh,vbll);
                        
                        
                        //store
                        _mm256_storeu_pd( &chh[i], vchh );
                        _mm256_storeu_pd( &chl[i], vchl );
                        _mm256_storeu_pd( &clh[i], vclh );
                        _mm256_storeu_pd( &cll[i], vcll );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _d_mul_qd(&schh,&schl,&sclh,&scll,sahh,sbhh,sbhl,sblh,sbll);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll;
                    sahh = ahh[0];
                    vahh = _mm256_broadcast_sd(&ahh[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _d_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vahh,vbhh,vbhl,vblh,vbll);
                        
                        
                        //store
                        _mm256_storeu_pd( &chh[i], vchh );
                        _mm256_storeu_pd( &chl[i], vchl );
                        _mm256_storeu_pd( &clh[i], vclh );
                        _mm256_storeu_pd( &cll[i], vcll );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _d_mul_qd(&schh,&schl,&sclh,&scll,sahh,sbhh,sbhl,sblh,sbll);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
            }
    }
}
/**
 *
 * @fn _d_dot_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the inner product of double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh input (double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _d_dot_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads];
                    double tchl [num_threads];
                    double tclh [num_threads];
                    double tcll [num_threads];
                    for (i = 0; i < num_threads; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sahh = ahh[i];
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _d_mul_qd(&schh,&schl,&sclh,&scll,sahh,sbhh,sbhl,sblh,sbll);
                        _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[tid],tchl[tid],tclh[tid],tcll[tid],schh,schl,sclh,scll);
                        
                        //store
                        tchh[tid] = szhh;
                        tchl[tid] = szhl;
                        tclh[tid] = szlh;
                        tcll[tid] = szll;
                    }
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[tid],tchl[tid],tclh[tid],tcll[tid],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads*4];
                    double tchl [num_threads*4];
                    double tclh [num_threads*4];
                    double tcll [num_threads*4];
                    for (i = 0; i < num_threads*4; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vzhh,vzhl,vzlh,vzll;
                    __m256d vchh = _mm256_setzero_pd();
                    __m256d vchl = _mm256_setzero_pd();
                    __m256d vclh = _mm256_setzero_pd();
                    __m256d vcll = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vahh = _mm256_set_pd(ahh[i],0,0,0);
                        vbhh = _mm256_set_pd(bhh[i],0,0,0);
                        vbhl = _mm256_set_pd(bhl[i],0,0,0);
                        vblh = _mm256_set_pd(blh[i],0,0,0);
                        vbll = _mm256_set_pd(bll[i],0,0,0);
                        
                        //calc
                        _d_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vbhh,vbhl,vblh,vbll);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        
                    }
                    //store
                    _mm256_storeu_pd( &tchh[4*tid], vchh );
                    _mm256_storeu_pd( &tchl[4*tid], vchl );
                    _mm256_storeu_pd( &tclh[4*tid], vclh );
                    _mm256_storeu_pd( &tcll[4*tid], vcll );
                    
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid+3],tchl[4*tid+3],tclh[4*tid+3],tcll[4*tid+3],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads*4];
                    double tchl [num_threads*4];
                    double tclh [num_threads*4];
                    double tcll [num_threads*4];
                    for (i = 0; i < num_threads*4; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vzhh,vzhl,vzlh,vzll;
                    __m256d vchh = _mm256_setzero_pd();
                    __m256d vchl = _mm256_setzero_pd();
                    __m256d vclh = _mm256_setzero_pd();
                    __m256d vcll = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahh = _mm256_loadu_pd(&ahh[i]);
                        
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _d_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vahh,vbhh,vbhl,vblh,vbll);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        
                    }
                    //store
                    _mm256_storeu_pd( &tchh[4*tid], vchh );
                    _mm256_storeu_pd( &tchl[4*tid], vchl );
                    _mm256_storeu_pd( &tclh[4*tid], vclh );
                    _mm256_storeu_pd( &tcll[4*tid], vcll );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        
                        //load
                        sahh = ahh[i];
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _d_mul_qd(&schh,&schl,&sclh,&scll,sahh,sbhh,sbhl,sblh,sbll);
                        _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],schh,schl,sclh,scll);
                        
                        //store
                        tchh[4*tid] = szhh;
                        tchl[4*tid] = szhl;
                        tclh[4*tid] = szlh;
                        tcll[4*tid] = szll;
                        
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid+i],tchl[4*tid+i],tclh[4*tid+i],tcll[4*tid+i],tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid]);
                        
                        //store
                        tchh[4*tid] = schh;
                        tchl[4*tid] = schl;
                        tclh[4*tid] = sclh;
                        tcll[4*tid] = scll;
                    }
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads*4];
                    double tchl [num_threads*4];
                    double tclh [num_threads*4];
                    double tcll [num_threads*4];
                    for (i = 0; i < num_threads*4; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vzhh,vzhl,vzlh,vzll;
                    __m256d vchh = _mm256_setzero_pd();
                    __m256d vchl = _mm256_setzero_pd();
                    __m256d vclh = _mm256_setzero_pd();
                    __m256d vcll = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahh = _mm256_loadu_pd(&ahh[i]);
                        
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _d_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vbhh,vbhl,vblh,vbll);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        
                    }
                    //store
                    _mm256_storeu_pd( &tchh[4*tid], vchh );
                    _mm256_storeu_pd( &tchl[4*tid], vchl );
                    _mm256_storeu_pd( &tclh[4*tid], vclh );
                    _mm256_storeu_pd( &tcll[4*tid], vcll );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        
                        //load
                        sahh = ahh[i];
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _d_mul_qd(&schh,&schl,&sclh,&scll,sahh,sbhh,sbhl,sblh,sbll);
                        _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],schh,schl,sclh,scll);
                        
                        //store
                        tchh[4*tid] = szhh;
                        tchl[4*tid] = szhl;
                        tclh[4*tid] = szlh;
                        tcll[4*tid] = szll;
                        
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid+i],tchl[4*tid+i],tclh[4*tid+i],tcll[4*tid+i],tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid]);
                        
                        //store
                        tchh[4*tid] = schh;
                        tchl[4*tid] = schl;
                        tclh[4*tid] = sclh;
                        tcll[4*tid] = scll;
                    }
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                    }
                }
                break;
            }
    }
}
/**
 *
 * @fn _d_mv_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-vector multiplicaion of double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh input (double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _d_mv_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
                    
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        sbhh = bhh[k];
                        sbhl = bhl[k];
                        sblh = blh[k];
                        sbll = bll[k];
                        for (i = 0; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            
                            //calc
                            _d_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                            
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int k,i;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double vte1[4]={0,0,0,0};
                    double vte2[4]={0,0,0,0};
                    double vte3[4]={0,0,0,0};
                    double vte4[4]={0,0,0,0};
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double schh,schl,sclh,scll;
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_set_pd(bhh[k],0,0,0);
                        vbhl = _mm256_set_pd(bhl[k],0,0,0);
                        vblh = _mm256_set_pd(blh[k],0,0,0);
                        vbll = _mm256_set_pd(bll[k],0,0,0);
                        for (i = 0; i < m; i++) {
                            //load
                            vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                            vzhh = _mm256_set_pd(tchh[m*tid+i],0,0,0);
                            vzhl = _mm256_set_pd(tchl[m*tid+i],0,0,0);
                            vzlh = _mm256_set_pd(tclh[m*tid+i],0,0,0);
                            vzll = _mm256_set_pd(tcll[m*tid+i],0,0,0);
                            
                            //calc
                            _d_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vahh,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            //store
                            _mm256_storeu_pd( &vte1[0], vchh );
                            _mm256_storeu_pd( &vte2[0], vchl );
                            _mm256_storeu_pd( &vte3[0], vclh );
                            _mm256_storeu_pd( &vte4[0], vcll );
                            
                            tchh[m*tid+i]=vte1[3];
                            tchl[m*tid+i]=vte2[3];
                            tclh[m*tid+i]=vte3[3];
                            tcll[m*tid+i]=vte4[3];
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_broadcast_sd(&bhh[k]);
                        vbhl = _mm256_broadcast_sd(&bhl[k]);
                        vblh = _mm256_broadcast_sd(&blh[k]);
                        vbll = _mm256_broadcast_sd(&bll[k]);
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vzhh = _mm256_loadu_pd(&tchh[m*tid+i]);
                            vzhl = _mm256_loadu_pd(&tchl[m*tid+i]);
                            vzlh = _mm256_loadu_pd(&tclh[m*tid+i]);
                            vzll = _mm256_loadu_pd(&tcll[m*tid+i]);
                            
                            //calc
                            _d_mul_qd_avx2(&vchh,&vchl,&vclh,&vcll,vahh,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            
                            //store
                            _mm256_storeu_pd( &tchh[m*tid+i], vchh );
                            _mm256_storeu_pd( &tchl[m*tid+i], vchl );
                            _mm256_storeu_pd( &tclh[m*tid+i], vclh );
                            _mm256_storeu_pd( &tcll[m*tid+i], vcll );
                        }
                        sbhh = bhh[k];
                        sbhl = bhl[k];
                        sblh = blh[k];
                        sbll = bll[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            
                            //calc
                            _d_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_broadcast_sd(&bhh[k]);
                        vbhl = _mm256_broadcast_sd(&bhl[k]);
                        vblh = _mm256_broadcast_sd(&blh[k]);
                        vbll = _mm256_broadcast_sd(&bll[k]);
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vzhh = _mm256_loadu_pd(&tchh[m*tid+i]);
                            vzhl = _mm256_loadu_pd(&tchl[m*tid+i]);
                            vzlh = _mm256_loadu_pd(&tclh[m*tid+i]);
                            vzll = _mm256_loadu_pd(&tcll[m*tid+i]);
                            
                            //calc
                            _d_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vahh,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            
                            //store
                            _mm256_storeu_pd( &tchh[m*tid+i], vchh );
                            _mm256_storeu_pd( &tchl[m*tid+i], vchl );
                            _mm256_storeu_pd( &tclh[m*tid+i], vclh );
                            _mm256_storeu_pd( &tcll[m*tid+i], vcll );
                        }
                        sbhh = bhh[k];
                        sbhl = bhl[k];
                        sblh = blh[k];
                        sbll = bll[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            
                            //calc
                            _d_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                    
                }
                
                break;
            }
            break;
    }
}
/**
 *
 * @fn _d_tmv_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the transposed matrix-vector multiplicaion of double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh input (double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _d_tmv_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    double schh,schl,sclh,scll,sahh,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        schh=0;
                        schl=0;
                        sclh=0;
                        scll=0;
                        for (i = 0; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sbhh = bhh[i];
                            sbhl = bhl[i];
                            sblh = blh[i];
                            sbll = bll[i];
                            
                            //calc
                            _d_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,schh,schl,sclh,scll);
                            
                            
                            //store
                            schh = szhh;
                            schl = szhl;
                            sclh = szlh;
                            scll = szll;
                        }
                        chh[k] = schh;
                        chl[k] = schl;
                        clh[k] = sclh;
                        cll[k] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m; i++) {
                            //load
                            vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                            vbhh = _mm256_set_pd(bhh[i],0,0,0);
                            vbhl = _mm256_set_pd(bhl[i],0,0,0);
                            vblh = _mm256_set_pd(blh[i],0,0,0);
                            vbll = _mm256_set_pd(bll[i],0,0,0);
                            
                            //calc
                            _d_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        
                        chh[k] = tchh[3];
                        chl[k] = tchl[3];
                        clh[k] = tclh[3];
                        cll[k] = tcll[3];
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    double schh,schl,sclh,scll,sahh,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll;
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vbhh = _mm256_loadu_pd(&bhh[i]);
                            vbhl = _mm256_loadu_pd(&bhl[i]);
                            vblh = _mm256_loadu_pd(&blh[i]);
                            vbll = _mm256_loadu_pd(&bll[i]);
                            
                            //calc
                            _d_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vahh,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sbhh = bhh[i];
                            sbhl = bhl[i];
                            sblh = blh[i];
                            sbll = bll[i];
                            
                            //calc
                            _d_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        for(i=1;i<4;i++){
                            //calc
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[i],tchl[i],tclh[i],tcll[i],tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        chh[k] = tchh[0];
                        chl[k] = tchl[0];
                        clh[k] = tclh[0];
                        cll[k] = tcll[0];
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    double schh,schl,sclh,scll,sahh,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll;
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vbhh = _mm256_loadu_pd(&bhh[i]);
                            vbhl = _mm256_loadu_pd(&bhl[i]);
                            vblh = _mm256_loadu_pd(&blh[i]);
                            vbll = _mm256_loadu_pd(&bll[i]);
                            
                            //calc
                            _d_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sbhh = bhh[i];
                            sbhl = bhl[i];
                            sblh = blh[i];
                            sbll = bll[i];
                            
                            //calc
                            _d_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        for(i=1;i<4;i++){
                            //calc
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[i],tchl[i],tclh[i],tcll[i],tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        chh[k] = tchh[0];
                        chl[k] = tchl[0];
                        clh[k] = tclh[0];
                        cll[k] = tcll[0];
                    }
                }
                break;
            }
            break;
    }
}
/**
 *
 * @fn _d_mm_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-matrix multiplicaion of double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh input (double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _d_mm_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    double sahh,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            sbhh = bhh[l*j+k];
                            sbhl = bhl[l*j+k];
                            sblh = blh[l*j+k];
                            sbll = bll[l*j+k];
                            for (i = 0; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                
                                //calc
                                _d_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sbhh,sbhl,sblh,sbll);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
                    double szhh[4]={0,0,0,0};
                    double szhl[4]={0,0,0,0};
                    double szlh[4]={0,0,0,0};
                    double szll[4]={0,0,0,0};
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_set_pd(bhh[l*j+k],0,0,0);
                            vbhl = _mm256_set_pd(bhl[l*j+k],0,0,0);
                            vblh = _mm256_set_pd(blh[l*j+k],0,0,0);
                            vbll = _mm256_set_pd(bll[l*j+k],0,0,0);
                            
                            for (i = 0; i < m; i++) {
                                vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                                
                                vchh = _mm256_set_pd(chh[m*j+i],0,0,0);
                                vchl = _mm256_set_pd(chl[m*j+i],0,0,0);
                                vclh = _mm256_set_pd(clh[m*j+i],0,0,0);
                                vcll = _mm256_set_pd(cll[m*j+i],0,0,0);
                                
                                //calc
                                _d_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vbhh,vbhl,vblh,vbll);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                _mm256_storeu_pd( &szhh[0], vchh );
                                _mm256_storeu_pd( &szhl[0], vchl );
                                _mm256_storeu_pd( &szlh[0], vclh );
                                _mm256_storeu_pd( &szll[0], vcll );
                                
                                chh[m*j+i] = szhh[3];
                                chl[m*j+i] = szhl[3];
                                clh[m*j+i] = szlh[3];
                                cll[m*j+i] = szll[3];
                            }
                        }
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    int p=m%4;
                    double sahh,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_broadcast_sd(&bhh[l*j+k]);
                            vbhl = _mm256_broadcast_sd(&bhl[l*j+k]);
                            vblh = _mm256_broadcast_sd(&blh[l*j+k]);
                            vbll = _mm256_broadcast_sd(&bll[l*j+k]);
                            
                            for (i = 0; i < m-p; i+=4) {
                                //load
                                vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                                vchh = _mm256_loadu_pd(&chh[m*j+i]);
                                vchl = _mm256_loadu_pd(&chl[m*j+i]);
                                vclh = _mm256_loadu_pd(&clh[m*j+i]);
                                vcll = _mm256_loadu_pd(&cll[m*j+i]);
                                
                                //calc
                                _d_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vahh,vbhh,vbhl,vblh,vbll);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                //store
                                _mm256_storeu_pd( &chh[m*j+i], vchh );
                                _mm256_storeu_pd( &chl[m*j+i], vchl );
                                _mm256_storeu_pd( &clh[m*j+i], vclh );
                                _mm256_storeu_pd( &cll[m*j+i], vcll );
                            }
                            sbhh = bhh[l*j+k];
                            sbhl = bhl[l*j+k];
                            sblh = blh[l*j+k];
                            sbll = bll[l*j+k];
                            for (i = m-p; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                
                                //calc
                                _d_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sbhh,sbhl,sblh,sbll);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    int p=m%4;
                    double sahh,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_broadcast_sd(&bhh[l*j+k]);
                            vbhl = _mm256_broadcast_sd(&bhl[l*j+k]);
                            vblh = _mm256_broadcast_sd(&blh[l*j+k]);
                            vbll = _mm256_broadcast_sd(&bll[l*j+k]);
                            
                            for (i = 0; i < m-p; i+=4) {
                                //load
                                vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                                vchh = _mm256_loadu_pd(&chh[m*j+i]);
                                vchl = _mm256_loadu_pd(&chl[m*j+i]);
                                vclh = _mm256_loadu_pd(&clh[m*j+i]);
                                vcll = _mm256_loadu_pd(&cll[m*j+i]);
                                
                                //calc
                                _d_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vbhh,vbhl,vblh,vbll);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                //store
                                _mm256_storeu_pd( &chh[m*j+i], vchh );
                                _mm256_storeu_pd( &chl[m*j+i], vchl );
                                _mm256_storeu_pd( &clh[m*j+i], vclh );
                                _mm256_storeu_pd( &cll[m*j+i], vcll );
                            }
                            sbhh = bhh[l*j+k];
                            sbhl = bhl[l*j+k];
                            sblh = blh[l*j+k];
                            sbll = bll[l*j+k];
                            for (i = m-p; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                
                                //calc
                                _d_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sbhh,sbhl,sblh,sbll);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
            }
            break;
    }
}

//-----------------------------------------------------------------
// Double-Double - Double operation
//-----------------------------------------------------------------

/**
 *
 * @fn _dd_scl_d(double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the scaler of double-double and double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi input (double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_scl_d(double *chi, double *clo, double *ahi, double *alo, double *bhi, int m, int n, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,salo,sbhi,schi,sclo;
                    sahi = ahi[0];
                    salo = alo[0];
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sbhi = bhi[i];
                        
                        //calc
                        _d_mul_dd(&schi,&sclo,sbhi,sahi,salo);
                        
                        //store
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahi,valo,vbhi,vchi,vclo,vch,vcl;
                    double vzhi[4]={0,0,0,0};
                    double vzlo[4]={0,0,0,0};
                    vahi = _mm256_set_pd(ahi[0],0,0,0);
                    valo = _mm256_set_pd(alo[0],0,0,0);
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vbhi = _mm256_set_pd(bhi[i],0,0,0);
                        
                        //calc
                        _d_mul_dd_fma(&vchi,&vclo,vbhi,vahi,valo);
                        
                        _mm256_storeu_pd( &vzhi[0], vchi );
                        _mm256_storeu_pd( &vzlo[0], vclo );
                        
                        chi[i]=vzhi[3];
                        clo[i]=vzlo[3];
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,salo,sbhi,schi,sclo;
                    
                    __m256d vahi,valo,vbhi,vchi,vclo,vch,vcl;
                    sahi = ahi[0];
                    salo = alo[0];
                    vahi = _mm256_broadcast_sd(&ahi[0]);
                    valo = _mm256_broadcast_sd(&alo[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhi = _mm256_loadu_pd(&bhi[i]);
                        
                        //calc
                        _d_mul_dd_avx2(&vchi,&vclo,vbhi,vahi,valo);
                        
                        //store
                        _mm256_storeu_pd( &chi[i], vchi );
                        _mm256_storeu_pd( &clo[i], vclo );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhi = bhi[i];
                        
                        //calc
                        _d_mul_dd(&schi,&sclo,sbhi,sahi,salo);
                        
                        //store
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,salo,sbhi,schi,sclo;
                    
                    __m256d vahi,valo,vbhi,vchi,vclo,vch,vcl;
                    sahi = ahi[0];
                    salo = alo[0];
                    vahi = _mm256_broadcast_sd(&ahi[0]);
                    valo = _mm256_broadcast_sd(&alo[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhi = _mm256_loadu_pd(&bhi[i]);
                        
                        //calc
                        _d_mul_dd_fma(&vchi,&vclo,vbhi,vahi,valo);
                        
                        //store
                        _mm256_storeu_pd( &chi[i], vchi );
                        _mm256_storeu_pd( &clo[i], vclo );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhi = bhi[i];
                        
                        //calc
                        _d_mul_dd(&schi,&sclo,sbhi,sahi,salo);
                        
                        //store
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
            }
    }
}
/**
 *
 * @fn _dd_mv_d(double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-vector multiplication of double-double and double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi input (double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_mv_d(double *chi, double *clo, double *ahi, double *alo,double *bhi, int m, int n, int l, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int k;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        ch[k]=0;
                        cl[k]=0;
                    }
                    double sahi,salo,sbhi,schi,sclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k ++) {
                        //load
                        sbhi = bhi[k];
                        for (i = 0; i < m; i ++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            
                            //calc
                            _d_mul_dd(&schi,&sclo,sbhi,sahi,salo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i = 0; i < m; i ++){
                        
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double zhi[4] = {0,0,0,0};
                    double zlo[4] = {0,0,0,0};
                    double ch [num_threads * m];
                    double cl [num_threads * m];
                    for (k = 0; k < num_threads * m; k ++) {
                        ch[k] = 0;
                        cl[k] = 0;
                    }
                    double sahi,salo,sbhi,schi,sclo;
                    __m256d vahi,valo,vbhi,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        //load
                        vbhi = _mm256_set_pd(bhi[k],0,0,0);
                        
                        for (i = 0; i < m; i++) {
                            //load
                            vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                            valo = _mm256_set_pd(alo[m*k+i],0,0,0);
                            vchi = _mm256_set_pd(ch[m*tid+i],0,0,0);
                            vclo = _mm256_set_pd(cl[m*tid+i],0,0,0);
                            
                            //calc
                            _d_mul_dd_fma(&vzhi,&vzlo,vbhi,vahi,valo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &zhi[0], vchi );
                            _mm256_storeu_pd( &zlo[0], vclo );
                            
                            ch[m*tid+i]=zhi[3];
                            cl[m*tid+i]=zlo[3];
                        }
                    }
#pragma omp critical
                    for(i = 0; i < m; i ++){
                        
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k, i, j;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    for (k = 0; k < num_threads * m; k++) {
                        ch[k] = 0;
                        cl[k] = 0;
                    }
                    double sahi,salo,sbhi,schi,sclo;
                    __m256d vahi,valo,vbhi,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhi = _mm256_broadcast_sd(&bhi[k]);
                        
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vchi = _mm256_loadu_pd(&ch[m*tid+i]);
                            vclo = _mm256_loadu_pd(&cl[m*tid+i]);
                            
                            //calc
                            _d_mul_dd_avx2(&vzhi,&vzlo,vbhi,vahi,valo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &ch[m*tid+i], vchi );
                            _mm256_storeu_pd( &cl[m*tid+i], vclo );
                            
                        }
                            sbhi = bhi[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            
                            //calc
                            _d_mul_dd(&schi,&sclo,sbhi,sahi,salo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k, i, j;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        ch[k]=0;
                        cl[k]=0;
                    }
                    double sahi,salo,sbhi,schi,sclo;
                    __m256d vahi,vbhi,valo,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhi = _mm256_broadcast_sd(&bhi[k]);
                        
                        for (i = 0; i < m-p; i+=4) {
                            
                            //load
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vchi = _mm256_loadu_pd(&ch[m*tid+i]);
                            vclo = _mm256_loadu_pd(&cl[m*tid+i]);
                            
                            //calc
                            _d_mul_dd_fma(&vzhi,&vzlo,vbhi,vahi,valo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &ch[m*tid+i], vchi );
                            _mm256_storeu_pd( &cl[m*tid+i], vclo );
                            
                        }
                            sbhi = bhi[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            
                            //calc
                            _d_mul_dd(&schi,&sclo,sbhi,sahi,salo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
            }
            break;
    }
}
/**
 *
 * @fn _dd_tmv_d(double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the transposed matrix-vector multiplication of double-double and double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi input (double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_tmv_d( double *chi, double *clo, double *ahi,double *alo,double *bhi,int m, int n, int l, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int k;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i;
                    double sahi,salo,sbhi,schi,sclo,ch,cl;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        ch=0;
                        cl=0;
                        for (i = 0; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            sbhi = bhi[i];
                            //calc
                            _d_mul_dd(&schi,&sclo,sbhi,sahi,salo);
                            _dd_add_dd(&ch,&cl,schi,sclo,ch,cl);
                            
                        }
                        chi[k]=ch;
                        clo[k]=cl;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int k,i;
                    __m256d vahi,valo,vbhi,vchi,vclo,vzhi,vzlo;
                    double zhi[4]={0,0,0,0};
                    double zlo[4]={0,0,0,0};
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vchi = _mm256_set_pd(0,0,0,0);
                        vclo = _mm256_set_pd(0,0,0,0);
                        
                        for (i = 0; i < m; i++) {
                            //load
                            vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                            valo = _mm256_set_pd(alo[m*k+i],0,0,0);
                            vbhi = _mm256_set_pd(bhi[i],0,0,0);
                            
                            //calc
                            _d_mul_dd_fma(&vzhi,&vzlo,vbhi,vahi,valo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                        }
                        _mm256_storeu_pd( &zhi[0], vchi );
                        _mm256_storeu_pd( &zlo[0], vclo );
                        
                        chi[k]=zhi[3];
                        clo[k]=zlo[3];
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k;
                    double ch[4]={0,0,0,0};
                    double cl[4]={0,0,0,0};
                    double sahi,salo,sbhi,schi,sclo;
                    __m256d vahi,valo,vbhi,vzhi,vzlo,vchi,vclo,vch,vcl;
                    int i,j;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vch = _mm256_set_pd(0,0,0,0);
                        vcl = _mm256_set_pd(0,0,0,0);
                        
                        for (i = 0; i < m-p; i+=4) {
                            
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vbhi = _mm256_loadu_pd(&bhi[i]);
                            
                            //calc
                            _d_mul_dd_avx2(&vzhi,&vzlo,vbhi,vahi,valo);
                            _dd_add_dd_avx2(&vch,&vcl,vzhi,vzlo,vch,vcl);
                        }
                        _mm256_storeu_pd( &ch[0], vch );
                        _mm256_storeu_pd( &cl[0], vcl );
                        
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            sbhi = bhi[i];
                            
                            //calc
                            _d_mul_dd(&schi,&sclo,sbhi,sahi,salo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[0],cl[0]);
                            
                            
                            //store
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        
                        for(i=1;i<4;i++){
                            
                            _dd_add_dd(&schi,&sclo,ch[0],cl[0],ch[i],cl[i]);
                            
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        chi[k]=ch[0];
                        clo[k]=cl[0];
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k;
                    double ch[4]={0,0,0,0};
                    double cl[4]={0,0,0,0};
                    double sahi,sbhi,salo,schi,sclo;
                    __m256d vahi,vbhi,valo,vzhi,vzlo,vchi,vclo,vch,vcl;
                    int i,j;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vch = _mm256_set_pd(0,0,0,0);
                        vcl = _mm256_set_pd(0,0,0,0);
                        
                        for (i = 0; i < m-p; i+=4) {
                            
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vbhi = _mm256_loadu_pd(&bhi[i]);
                            
                            //calc
                            _d_mul_dd_fma(&vzhi,&vzlo,vbhi,vahi,valo);
                            _dd_add_dd_avx2(&vch,&vcl,vzhi,vzlo,vch,vcl);
                        }
                        _mm256_storeu_pd( &ch[0], vch );
                        _mm256_storeu_pd( &cl[0], vcl );
                        
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];                            
                            sbhi = bhi[i];
                            
                            //calc
                            _d_mul_dd(&schi,&sclo,sbhi,sahi,salo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[0],cl[0]);
                            
                            
                            //store
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        
                        for(i=1;i<4;i++){
                            
                            _dd_add_dd(&schi,&sclo,ch[0],cl[0],ch[i],cl[i]);
                            
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        chi[k]=ch[0];
                        clo[k]=cl[0];
                    }
                }
                break;
            }
            break;
    }
}
/**
 *
 * @fn _dd_mm_d(double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-matrix multiplication of double-double and double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi input (double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_mm_d(double *chi, double *clo, double *ahi, double *alo,double *bhi,int m, int n, int l, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int k;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,sbhi,salo,schi,sclo;
                    int i,j,k;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            for (i = 0; i < m; i++) {
                                //load
                                sahi = ahi[m*k+i];
                                salo = alo[m*k+i];
                                sbhi = bhi[l*j+k];
 
                                
                                //calc
                                _d_mul_dd(&schi,&sclo,sbhi,sahi,salo);
                                _dd_add_dd(&schi,&sclo,schi,sclo,chi[m*j+i],clo[m*j+i]);
                                
                                //store
                                chi[m*j+i] = schi;
                                clo[m*j+i] = sclo;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahi,vbhi,valo,vchi,vclo,vzhi,vzlo;
                    double zhi[4]={0,0,0,0};
                    double zlo[4]={0,0,0,0};
                    int j,k,i;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhi = _mm256_set_pd(bhi[l*j+k],0,0,0);
                            for (i = 0; i < m; i++) {
                                vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                                valo = _mm256_set_pd(alo[m*k+i],0,0,0);
                                vchi = _mm256_set_pd(chi[m*j+i],0,0,0);
                                vclo = _mm256_set_pd(clo[m*j+i],0,0,0);
                                
                                //calc
                                _d_mul_dd_fma(&vzhi,&vzlo,vbhi,vahi,valo);
                                _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                                
                                _mm256_storeu_pd( &zhi[0], vchi );
                                _mm256_storeu_pd( &zlo[0], vclo );
                                
                                chi[m*j+i]=zhi[3];
                                clo[m*j+i]=zlo[3];
                            }
                        }
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    int p=m%4;
                    double sahi,sbhi,salo,schi,sclo;
                    __m256d vahi,vbhi,valo,vzhi,vzlo,vchi,vclo,vch,vcl;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhi = _mm256_broadcast_sd(&bhi[l*j+k]);
                            for (i = 0; i < m-p; i+=4) {
                                vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                                valo = _mm256_loadu_pd(&alo[m*k+i]);
                                vchi = _mm256_loadu_pd(&chi[m*j+i]);
                                vclo = _mm256_loadu_pd(&clo[m*j+i]);
                                
                                //calc
                                _d_mul_dd_avx2(&vzhi,&vzlo,vbhi,vahi,valo);
                                _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                                
                                _mm256_storeu_pd( &chi[m*j+i], vchi );
                                _mm256_storeu_pd( &clo[m*j+i], vclo );
                            }
                            for (i = m-p; i < m; i++) {
                                //load
                                sahi = ahi[m*k+i];
                                salo = alo[m*k+i];
                                sbhi = bhi[l*j+k];
                                
                                //calc
                                _d_mul_dd(&schi,&sclo,sbhi,sahi,salo);
                                _dd_add_dd(&schi,&sclo,schi,sclo,chi[m*j+i],clo[m*j+i]);
                                
                                //store
                                chi[m*j+i] = schi;
                                clo[m*j+i] = sclo;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    int p=m%4;
                    double sahi,sbhi,salo,schi,sclo;
                    __m256d vahi,vbhi,valo,vzhi,vzlo,vchi,vclo,vch,vcl;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhi = _mm256_broadcast_sd(&bhi[l*j+k]);
                            for (i = 0; i < m-p; i+=4) {
                                vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                                valo = _mm256_loadu_pd(&alo[m*k+i]);
                                vchi = _mm256_loadu_pd(&chi[m*j+i]);
                                vclo = _mm256_loadu_pd(&clo[m*j+i]);
                                
                                //calc
                                _d_mul_dd_fma(&vzhi,&vzlo,vbhi,vahi,valo);
                                _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                                
                                _mm256_storeu_pd( &chi[m*j+i], vchi );
                                _mm256_storeu_pd( &clo[m*j+i], vclo );
                            }
                            for (i = m-p; i < m; i++) {
                                //load
                                sahi = ahi[m*k+i];
                                salo = alo[m*k+i];
                                sbhi = bhi[l*j+k];
                                
                                //calc
                                _d_mul_dd(&schi,&sclo,sbhi,sahi,salo);
                                _dd_add_dd(&schi,&sclo,schi,sclo,chi[m*j+i],clo[m*j+i]);
                                
                                //store
                                chi[m*j+i] = schi;
                                clo[m*j+i] = sclo;
                            }
                        }
                    }
                }
                break;
            }
            break;
    }
}
//-----------------------------------------------------------------
// Double-Double - Double-Double operation
//-----------------------------------------------------------------


/**
 *
 * @fn _dd_a_dd(double*,double*,double*,double*,double*,double*,int,int,int,int)
 * @brief You can compute the addition of double-double and double-double numbers in a vector or matrix. You can choose the number of OpenMP threads and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _dd_a_dd(double *chi, double *clo, double *ahi, double *alo, double *bhi, double *blo, int m, int n, int omp_threadNum, int avx)
{
    int i;
    int p = m % 4;
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    switch(avx){
        case 0:
#pragma omp parallel num_threads(omp_threadNum)
        {
            double sahi,salo,sbhi,sblo,schi,sclo;
#pragma omp for
            for (i = 0; i < m * n; i ++){
                //load
                sahi = ahi[i];
                salo = alo[i];
                sbhi = bhi[i];
                sblo = blo[i];
                
                //calc
                _dd_add_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                
                //store
                chi[i] = schi;
                clo[i] = sclo;
            }
        }
        break;
        case 1:
#pragma omp parallel num_threads(omp_threadNum)
        {
            double sahi,salo,sbhi,sblo,schi,sclo;
            __m256d vahi,valo,vbhi,vblo,vchi,vclo;
#pragma omp for
            for (i = 0; i < m * n - p; i += 4){
                //load
                vahi = _mm256_loadu_pd(&ahi[i]);
                valo = _mm256_loadu_pd(&alo[i]);
                vbhi = _mm256_loadu_pd(&bhi[i]);
                vblo = _mm256_loadu_pd(&blo[i]);
                
                //calc
                _dd_add_dd_avx2(&vchi,&vclo,vahi,valo,vbhi,vblo);
                
                //store
                _mm256_storeu_pd( &chi[i], vchi );
                _mm256_storeu_pd( &clo[i], vclo );
            }
#pragma omp for
            for (i = m * n - p; i < m * n; i ++){
                //load
                sahi = ahi[i];
                salo = alo[i];
                sbhi = bhi[i];
                sblo = blo[i];
                
                //calc
                _dd_add_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                
                //store
                chi[i] = schi;
                clo[i] = sclo;
            }
        }
        break;
        
    }
}

/**
 *
 * @fn _dd_a_dd_ieee(double*,double*,double*,double*,double*,double*,int,int,int,int)
 * @brief You can compute the addition of double-double and double-double numbers in a vector or matrix. You can choose the number of OpenMP threads and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _dd_a_dd_ieee(double *chi, double *clo, double *ahi, double *alo, double *bhi, double *blo, int m, int n, int omp_threadNum, int avx)
{
    int i;
    int p = m % 4;
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    switch(avx){
        case 0:
#pragma omp parallel num_threads(omp_threadNum)
        {
            double sahi,salo,sbhi,sblo,schi,sclo;
#pragma omp for
            for (i = 0; i < m * n; i ++){
                //load
                sahi = ahi[i];
                salo = alo[i];
                sbhi = bhi[i];
                sblo = blo[i];
                
                //calc
                _dd_add_dd_ieee(&schi,&sclo,sahi,salo,sbhi,sblo);
                
                //store
                chi[i] = schi;
                clo[i] = sclo;
            }
        }
        break;
        case 1:
#pragma omp parallel num_threads(omp_threadNum)
        {
            double sahi,salo,sbhi,sblo,schi,sclo;
            __m256d vahi,valo,vbhi,vblo,vchi,vclo;
#pragma omp for
            for (i = 0; i < m * n - p; i += 4){
                //load
                vahi = _mm256_loadu_pd(&ahi[i]);
                valo = _mm256_loadu_pd(&alo[i]);
                vbhi = _mm256_loadu_pd(&bhi[i]);
                vblo = _mm256_loadu_pd(&blo[i]);
                
                //calc
                _dd_add_dd_avx2_ieee(&vchi,&vclo,vahi,valo,vbhi,vblo);
                
                //store
                _mm256_storeu_pd( &chi[i], vchi );
                _mm256_storeu_pd( &clo[i], vclo );
            }
#pragma omp for
            for (i = m * n - p; i < m * n; i ++){
                //load
                sahi = ahi[i];
                salo = alo[i];
                sbhi = bhi[i];
                sblo = blo[i];
                
                //calc
                _dd_add_dd_ieee(&schi,&sclo,sahi,salo,sbhi,sblo);
                
                //store
                chi[i] = schi;
                clo[i] = sclo;
            }
        }
        break;
        
    }
}

/**
 *
 * @fn _dd_scl_dd(double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the scaler of double-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_scl_dd(double *chi, double *clo, double *ahi,double *alo, double *bhi, double *blo,int m, int n, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    sahi = ahi[0];
                    salo = alo[0];
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                        
                        //store
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahi,valo,vbhi,vblo,vchi,vclo,vch,vcl;
                    double vzhi[4]={0,0,0,0};
                    double vzlo[4]={0,0,0,0};
                    vahi = _mm256_set_pd(ahi[0],0,0,0);
                    valo = _mm256_set_pd(alo[0],0,0,0);
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vbhi = _mm256_set_pd(bhi[i],0,0,0);
                        vblo = _mm256_set_pd(blo[i],0,0,0);
                        
                        //calc
                        _dd_mul_dd_fma(&vchi,&vclo,vahi,valo,vbhi,vblo);
                        
                        _mm256_storeu_pd( &vzhi[0], vchi );
                        _mm256_storeu_pd( &vzlo[0], vclo );
                        
                        chi[i]=vzhi[3];
                        clo[i]=vzlo[3];
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    
                    __m256d vahi,valo,vbhi,vblo,vchi,vclo,vch,vcl;
                    sahi = ahi[0];
                    salo = alo[0];
                    vahi = _mm256_broadcast_sd(&ahi[0]);
                    valo = _mm256_broadcast_sd(&alo[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhi = _mm256_loadu_pd(&bhi[i]);
                        vblo = _mm256_loadu_pd(&blo[i]);
                        
                        //calc
                        _dd_mul_dd_avx2(&vchi,&vclo,vahi,valo,vbhi,vblo);
                        
                        //store
                        _mm256_storeu_pd( &chi[i], vchi );
                        _mm256_storeu_pd( &clo[i], vclo );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                        
                        //store
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    
                    __m256d vahi,valo,vbhi,vblo,vchi,vclo,vch,vcl;
                    sahi = ahi[0];
                    salo = alo[0];
                    vahi = _mm256_broadcast_sd(&ahi[0]);
                    valo = _mm256_broadcast_sd(&alo[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhi = _mm256_loadu_pd(&bhi[i]);
                        vblo = _mm256_loadu_pd(&blo[i]);
                        
                        //calc
                        _dd_mul_dd_fma(&vchi,&vclo,vahi,valo,vbhi,vblo);
                        
                        //store
                        _mm256_storeu_pd( &chi[i], vchi );
                        _mm256_storeu_pd( &clo[i], vclo );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                        
                        //store
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
            }
    }
}

/**
 *
 * @fn _dd_dot_dd(double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the inner product of double-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_dot_dd(double *chi, double *clo, double *ahi, double *alo, double *bhi, double *blo,int m, int n, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahi,salo,sbhi,sblo,schi,sclo,szhi,szlo;
                    double ch [num_threads];
                    double cl [num_threads];
                    for (i = 0; i < num_threads; i ++){
                        ch[i]=0;
                        cl[i]=0;
                    }
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sahi = ahi[i];
                        salo = alo[i];
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                        _dd_add_dd(&szhi,&szlo,ch[tid],cl[tid],schi,sclo);
                        
                        //store
                        ch[tid] = szhi;
                        cl[tid] = szlo;
                    }
#pragma omp critical
                    {
                        //calc
                        _dd_add_dd(&schi,&sclo,ch[tid],cl[tid],chi[0],clo[0]);
                        
                        //store
                        chi[0] = schi;
                        clo[0] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double schi,sclo;
                    double ch [num_threads*4];
                    double cl [num_threads*4];
                    for (i = 0; i < num_threads * 4; i ++){
                        ch[i]=0;
                        cl[i]=0;
                    }
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo;
                    __m256d vchi = _mm256_setzero_pd();
                    __m256d vclo = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vahi = _mm256_set_pd(ahi[i],0,0,0);
                        valo = _mm256_set_pd(alo[i],0,0,0);
                        vbhi = _mm256_set_pd(bhi[i],0,0,0);
                        vblo = _mm256_set_pd(blo[i],0,0,0);
                        
                        //calc
                        _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                        _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                        
                    }
                    //store
                    _mm256_storeu_pd( &ch[4*tid], vchi );
                    _mm256_storeu_pd( &cl[4*tid], vclo );
                    
#pragma omp critical
                    {
                        //calc
                        _dd_add_dd(&schi,&sclo,ch[4*tid+3],cl[4*tid+3],chi[0],clo[0]);
                        
                        //store
                        chi[0] = schi;
                        clo[0] = sclo;
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahi,salo,sbhi,sblo,schi,sclo,szhi,szlo;
                    double ch [num_threads*4];
                    double cl [num_threads*4];
                    for (i = 0; i < num_threads * 4; i ++){
                        ch[i]=0;
                        cl[i]=0;
                    }
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo;
                    __m256d vchi = _mm256_setzero_pd();
                    __m256d vclo = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahi = _mm256_loadu_pd(&ahi[i]);
                        valo = _mm256_loadu_pd(&alo[i]);
                        vbhi = _mm256_loadu_pd(&bhi[i]);
                        vblo = _mm256_loadu_pd(&blo[i]);
                        
                        //calc
                        _dd_mul_dd_avx2(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                        _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                        
                    }
                    //store
                    _mm256_storeu_pd( &ch[4*tid], vchi );
                    _mm256_storeu_pd( &cl[4*tid], vclo );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        //load
                        sahi = ahi[i];
                        salo = alo[i];
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _dd_mul_dd(&szhi,&szlo,sahi,salo,sbhi,sblo);
                        _dd_add_dd(&schi,&sclo,szhi,szlo,ch[4*tid],cl[4*tid]);
                        
                        //store
                        ch[4*tid] = schi;
                        cl[4*tid] = sclo;
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _dd_add_dd(&schi,&sclo,ch[4*tid+i],cl[4*tid+i],ch[4*tid],cl[4*tid]);
                        
                        //store
                        ch[4*tid] = schi;
                        cl[4*tid] = sclo;
                    }
#pragma omp critical
                    {
                        //calc
                        _dd_add_dd(&schi,&sclo,ch[4*tid],cl[4*tid],chi[0],clo[0]);
                        
                        //store
                        chi[0] = schi;
                        clo[0] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahi,salo,sbhi,sblo,schi,sclo,szhi,szlo;
                    double ch [num_threads*4];
                    double cl [num_threads*4];
                    for (i = 0; i < num_threads * 4; i ++){
                        ch[i]=0;
                        cl[i]=0;
                    }
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo;
                    __m256d vchi = _mm256_setzero_pd();
                    __m256d vclo = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahi = _mm256_loadu_pd(&ahi[i]);
                        valo = _mm256_loadu_pd(&alo[i]);
                        vbhi = _mm256_loadu_pd(&bhi[i]);
                        vblo = _mm256_loadu_pd(&blo[i]);
                        
                        //calc
                        _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                        _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                        
                    }
                    //store
                    _mm256_storeu_pd( &ch[4*tid], vchi );
                    _mm256_storeu_pd( &cl[4*tid], vclo );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        //load
                        sahi = ahi[i];
                        salo = alo[i];
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _dd_mul_dd(&szhi,&szlo,sahi,salo,sbhi,sblo);
                        _dd_add_dd(&schi,&sclo,szhi,szlo,ch[4*tid],cl[4*tid]);
                        
                        //store
                        ch[4*tid] = schi;
                        cl[4*tid] = sclo;
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _dd_add_dd(&schi,&sclo,ch[4*tid+i],cl[4*tid+i],ch[4*tid],cl[4*tid]);
                        
                        //store
                        ch[4*tid] = schi;
                        cl[4*tid] = sclo;
                    }
#pragma omp critical
                    {
                        //calc
                        _dd_add_dd(&schi,&sclo,ch[4*tid],cl[4*tid],chi[0],clo[0]);
                        
                        //store
                        chi[0] = schi;
                        clo[0] = sclo;
                    }
                }
                break;
            }
    }
}

/**
 *
 * @fn _dd_dot_dd_ieee(double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the inner product of double-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_dot_dd_ieee(double *chi, double *clo, double *ahi, double *alo, double *bhi, double *blo,int m, int n, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahi,salo,sbhi,sblo,schi,sclo,szhi,szlo;
                    double ch [num_threads];
                    double cl [num_threads];
                    for (i = 0; i < num_threads; i ++){
                        ch[i]=0;
                        cl[i]=0;
                    }
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sahi = ahi[i];
                        salo = alo[i];
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                        _dd_add_dd_ieee(&szhi,&szlo,ch[tid],cl[tid],schi,sclo);
                        
                        //store
                        ch[tid] = szhi;
                        cl[tid] = szlo;
                    }
#pragma omp critical
                    {
                        //calc
                        _dd_add_dd_ieee(&schi,&sclo,ch[tid],cl[tid],chi[0],clo[0]);
                        
                        //store
                        chi[0] = schi;
                        clo[0] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double schi,sclo;
                    double ch [num_threads*4];
                    double cl [num_threads*4];
                    for (i = 0; i < num_threads * 4; i ++){
                        ch[i]=0;
                        cl[i]=0;
                    }
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo;
                    __m256d vchi = _mm256_setzero_pd();
                    __m256d vclo = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vahi = _mm256_set_pd(ahi[i],0,0,0);
                        valo = _mm256_set_pd(alo[i],0,0,0);
                        vbhi = _mm256_set_pd(bhi[i],0,0,0);
                        vblo = _mm256_set_pd(blo[i],0,0,0);
                        
                        //calc
                        _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                        _dd_add_dd_avx2_ieee(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                        
                    }
                    //store
                    _mm256_storeu_pd( &ch[4*tid], vchi );
                    _mm256_storeu_pd( &cl[4*tid], vclo );
                    
#pragma omp critical
                    {
                        //calc
                        _dd_add_dd_ieee(&schi,&sclo,ch[4*tid+3],cl[4*tid+3],chi[0],clo[0]);
                        
                        //store
                        chi[0] = schi;
                        clo[0] = sclo;
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahi,salo,sbhi,sblo,schi,sclo,szhi,szlo;
                    double ch [num_threads*4];
                    double cl [num_threads*4];
                    for (i = 0; i < num_threads * 4; i ++){
                        ch[i]=0;
                        cl[i]=0;
                    }
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo;
                    __m256d vchi = _mm256_setzero_pd();
                    __m256d vclo = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahi = _mm256_loadu_pd(&ahi[i]);
                        valo = _mm256_loadu_pd(&alo[i]);
                        vbhi = _mm256_loadu_pd(&bhi[i]);
                        vblo = _mm256_loadu_pd(&blo[i]);
                        
                        //calc
                        _dd_mul_dd_avx2(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                        _dd_add_dd_avx2_ieee(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                        
                    }
                    //store
                    _mm256_storeu_pd( &ch[4*tid], vchi );
                    _mm256_storeu_pd( &cl[4*tid], vclo );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        //load
                        sahi = ahi[i];
                        salo = alo[i];
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _dd_mul_dd(&szhi,&szlo,sahi,salo,sbhi,sblo);
                        _dd_add_dd_ieee(&schi,&sclo,szhi,szlo,ch[4*tid],cl[4*tid]);
                        
                        //store
                        ch[4*tid] = schi;
                        cl[4*tid] = sclo;
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _dd_add_dd_ieee(&schi,&sclo,ch[4*tid+i],cl[4*tid+i],ch[4*tid],cl[4*tid]);
                        
                        //store
                        ch[4*tid] = schi;
                        cl[4*tid] = sclo;
                    }
#pragma omp critical
                    {
                        //calc
                        _dd_add_dd_ieee(&schi,&sclo,ch[4*tid],cl[4*tid],chi[0],clo[0]);
                        
                        //store
                        chi[0] = schi;
                        clo[0] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahi,salo,sbhi,sblo,schi,sclo,szhi,szlo;
                    double ch [num_threads*4];
                    double cl [num_threads*4];
                    for (i = 0; i < num_threads * 4; i ++){
                        ch[i]=0;
                        cl[i]=0;
                    }
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo;
                    __m256d vchi = _mm256_setzero_pd();
                    __m256d vclo = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahi = _mm256_loadu_pd(&ahi[i]);
                        valo = _mm256_loadu_pd(&alo[i]);
                        vbhi = _mm256_loadu_pd(&bhi[i]);
                        vblo = _mm256_loadu_pd(&blo[i]);
                        
                        //calc
                        _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                        _dd_add_dd_avx2_ieee(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                        
                    }
                    //store
                    _mm256_storeu_pd( &ch[4*tid], vchi );
                    _mm256_storeu_pd( &cl[4*tid], vclo );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        //load
                        sahi = ahi[i];
                        salo = alo[i];
                        sbhi = bhi[i];
                        sblo = blo[i];
                        
                        //calc
                        _dd_mul_dd(&szhi,&szlo,sahi,salo,sbhi,sblo);
                        _dd_add_dd_ieee(&schi,&sclo,szhi,szlo,ch[4*tid],cl[4*tid]);
                        
                        //store
                        ch[4*tid] = schi;
                        cl[4*tid] = sclo;
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _dd_add_dd_ieee(&schi,&sclo,ch[4*tid+i],cl[4*tid+i],ch[4*tid],cl[4*tid]);
                        
                        //store
                        ch[4*tid] = schi;
                        cl[4*tid] = sclo;
                    }
#pragma omp critical
                    {
                        //calc
                        _dd_add_dd_ieee(&schi,&sclo,ch[4*tid],cl[4*tid],chi[0],clo[0]);
                        
                        //store
                        chi[0] = schi;
                        clo[0] = sclo;
                    }
                }
                break;
            }
    }
}

/**
 *
 * @fn _dd_mv_dd(double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-vector multiplication of double-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_mv_dd(double *chi, double *clo, double *ahi, double *alo, double *bhi, double *blo,int m, int n, int l, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int k;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        ch[k]=0;
                        cl[k]=0;
                    }
                    double sahi,salo,sbhi,sblo,schi,sclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k ++) {
                        //load
                        sbhi = bhi[k];
                        sblo = blo[k];
                        for (i = 0; i < m; i ++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i = 0; i < m; i ++){
                        
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double zhi[4] = {0,0,0,0};
                    double zlo[4] = {0,0,0,0};
                    double ch [num_threads * m];
                    double cl [num_threads * m];
                    for (k = 0; k < num_threads * m; k ++) {
                        ch[k] = 0;
                        cl[k] = 0;
                    }
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        //load
                        vbhi = _mm256_set_pd(bhi[k],0,0,0);
                        vblo = _mm256_set_pd(blo[k],0,0,0);
                        
                        for (i = 0; i < m; i++) {
                            //load
                            vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                            valo = _mm256_set_pd(alo[m*k+i],0,0,0);
                            vchi = _mm256_set_pd(ch[m*tid+i],0,0,0);
                            vclo = _mm256_set_pd(cl[m*tid+i],0,0,0);
                            
                            //calc
                            _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &zhi[0], vchi );
                            _mm256_storeu_pd( &zlo[0], vclo );
                            
                            ch[m*tid+i]=zhi[3];
                            cl[m*tid+i]=zlo[3];
                        }
                    }
#pragma omp critical
                    for(i = 0; i < m; i ++){
                        
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 2: // OMP to inner loop
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int k, i, j;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double zhi[4] = {0,0,0,0};
                    double zlo[4] = {0,0,0,0};
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo;
                    for (k = 0; k < l; k++) {
                        //load
                        vbhi = _mm256_set_pd(bhi[k],0,0,0);
                        vblo = _mm256_set_pd(blo[k],0,0,0);
#pragma omp for schedule(static)
                        for (i = 0; i < m; i++) {
                            
                            //load
                            vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                            valo = _mm256_set_pd(alo[m*k+i],0,0,0);
                            vchi = _mm256_set_pd(chi[i],0,0,0);
                            vclo = _mm256_set_pd(clo[i],0,0,0);
                            
                            //calc
                            _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
             
                            //store
                            _mm256_storeu_pd( &zhi[0], vchi );
                            _mm256_storeu_pd( &zlo[0], vclo );
                            
                            chi[i]=zhi[3];
                            clo[i]=zlo[3];
                        }
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case -2:
                    #pragma omp parallel num_threads(omp_threadNum)
        {
            int k;
            int num_threads=omp_get_num_threads();
            int tid = omp_get_thread_num();
            double ch [num_threads*m];
            double cl [num_threads*m];
            for (k = 0; k < num_threads*m; k++) {
                ch[k]=0;
                cl[k]=0;
            }
            __m256d vahi,valo,vbhi,vblo,vchi,vclo,vch,vcl;
            __m256d t=_mm256_set_pd(0,0,0,0);
            __m256d t1=_mm256_set_pd(0,0,0,0);
            __m256d ah=_mm256_set_pd(0,0,0,0);
            __m256d al=_mm256_set_pd(0,0,0,0);
            __m256d bh=_mm256_set_pd(0,0,0,0);
            __m256d bl=_mm256_set_pd(0,0,0,0);
            __m256d p1=_mm256_set_pd(0,0,0,0);
            __m256d p2=_mm256_set_pd(0,0,0,0);
            __m256d chi_n=_mm256_set_pd(0,0,0,0);
            __m256d clo_n=_mm256_set_pd(0,0,0,0);
            __m256d cons=_mm256_set_pd(134217729,134217729,134217729,134217729);
            int i,j;
#pragma omp for schedule(static)
            for (k = 0; k < l; k++) {
                vbhi = _mm256_broadcast_sd(&bhi[k]);
                vblo = _mm256_broadcast_sd(&blo[k]);
                
                for (i = 0; i < m; i+=4) {
                    
                    vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                    valo = _mm256_loadu_pd(&alo[m*k+i]);
                    vch = _mm256_loadu_pd(&ch[m*tid+i]);
                    vcl = _mm256_loadu_pd(&cl[m*tid+i]);
                    
                    //multiplication
                    
                    p1 = _mm256_mul_pd(vahi , vbhi);
                    p2 = _mm256_fmsub_pd(vahi , vbhi, p1);
                    
                    p2 = _mm256_fmadd_pd(vahi , vblo , p2);
                    p2 = _mm256_fmadd_pd(valo , vbhi , p2);
                    chi_n =_mm256_add_pd(p1 , p2);
                    clo_n = _mm256_sub_pd(p2, _mm256_sub_pd(chi_n, p1));
                    
                    //addition
                    ah = _mm256_add_pd(vch, chi_n);
                    t = _mm256_sub_pd(ah, vch);
                    bh = _mm256_add_pd(
                                       _mm256_sub_pd(vch,
                                                     _mm256_sub_pd(ah , t)),
                                       _mm256_sub_pd(chi_n , t));
                    
                    al = _mm256_add_pd(vcl , clo_n);
                    bh = _mm256_add_pd(bh , al);
                    vch = _mm256_add_pd(ah , bh);
                    vcl = _mm256_sub_pd(bh , _mm256_sub_pd(vch , ah));
                    
                    _mm256_storeu_pd( &ch[m*tid+i], vch );
                    _mm256_storeu_pd( &cl[m*tid+i], vcl );
                    
                }
            }
#pragma omp critical
            for(k=0;k<m;k+=4){
                vchi = _mm256_loadu_pd(&chi[k]);
                vclo = _mm256_loadu_pd(&clo[k]);
                vch = _mm256_loadu_pd(&ch[m*tid+k]);
                vcl = _mm256_loadu_pd(&cl[m*tid+k]);
                
                ah = _mm256_add_pd(vchi, vch);
                t = _mm256_sub_pd(ah, vchi);
                bh = _mm256_add_pd(
                                   _mm256_sub_pd(vchi,
                                                 _mm256_sub_pd(ah , t)),
                                   _mm256_sub_pd(vch , t));
                
                al = _mm256_add_pd(vclo , vcl);
                bh = _mm256_add_pd(bh , al);
                vchi = _mm256_add_pd(ah , bh);
                vclo = _mm256_sub_pd(bh , _mm256_sub_pd(vchi , ah));
                
                _mm256_storeu_pd( &chi[k], vchi );
                _mm256_storeu_pd( &clo[k], vclo );
            }
        }
                break;
                case -1: // (vi)
                    #pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k, i, j;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    double zhi [num_threads*4];
                    double zlo [num_threads*4];
                    for (k = 0; k < num_threads * m; k++) {
                        ch[k] = 0;
                        cl[k] = 0;
                    }
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l-p; k+=4) {
                        vbhi = _mm256_loadu_pd(&bhi[k]);
                        vblo = _mm256_loadu_pd(&blo[k]);
                        for (i = 0; i < m; i++) {
                            //load
                            vahi = _mm256_set_pd(ahi[m*(k+3)+i],ahi[m*(k+2)+i],ahi[m*(k+1)+i],ahi[m*k+i]);
                            valo = _mm256_set_pd(alo[m*(k+3)+i],alo[m*(k+2)+i],alo[m*(k+1)+i],alo[m*k+i]);
                            
                            //calc
                            _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);

                            //store
                            _mm256_storeu_pd( &zhi[tid*4], vzhi );
                            _mm256_storeu_pd( &zlo[tid*4], vzlo );

                            schi = zhi[tid];
                            sclo = zlo[tid];
                            for(j=1;j<4;j++){
                                //calc
                                _dd_add_dd(&schi,&sclo,zhi[4*tid+j],zlo[4*tid+j],schi,sclo);
                            }
                            _dd_add_dd(&schi,&sclo,ch[m*tid+i],cl[m*tid+i],schi,sclo);
                           
                                //store
                                ch[m*tid+i] = schi;
                                cl[m*tid+i] = sclo;
                        }
                    }
                    for (k = l-p; k < l; k++) {
                        sbhi = bhi[k];
                        sblo = blo[k];
                        for (i = 0; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            schi = ch[m*tid+i];
                            sclo = cl[m*tid+i];
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k, i, j;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    for (k = 0; k < num_threads * m; k++) {
                        ch[k] = 0;
                        cl[k] = 0;
                    }
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhi = _mm256_broadcast_sd(&bhi[k]);
                        vblo = _mm256_broadcast_sd(&blo[k]);
                        
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vchi = _mm256_loadu_pd(&ch[m*tid+i]);
                            vclo = _mm256_loadu_pd(&cl[m*tid+i]);
                            
                            //calc
                            _dd_mul_dd_avx2(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &ch[m*tid+i], vchi );
                            _mm256_storeu_pd( &cl[m*tid+i], vclo );
                            
                        }
                        sbhi = bhi[k];
                        sblo = blo[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k, i, j;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        ch[k]=0;
                        cl[k]=0;
                    }
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhi = _mm256_broadcast_sd(&bhi[k]);
                        vblo = _mm256_broadcast_sd(&blo[k]);
                        
                        for (i = 0; i < m-p; i+=4) {
                            
                            //load
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vchi = _mm256_loadu_pd(&ch[m*tid+i]);
                            vclo = _mm256_loadu_pd(&cl[m*tid+i]);
                            
                            //calc
                            _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &ch[m*tid+i], vchi );
                            _mm256_storeu_pd( &cl[m*tid+i], vclo );
                            
                        }
                        sbhi = bhi[k];
                        sblo = blo[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        _dd_add_dd(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 2: // OpenMP to inner loop
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k, i, j;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo;
                    for (k = 0; k < l; k++) {
                        vbhi = _mm256_broadcast_sd(&bhi[k]);
                        vblo = _mm256_broadcast_sd(&blo[k]);
#pragma omp for schedule(static)
                        for (i = 0; i < m-p; i+=4) {
                            
                            //load
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vchi = _mm256_loadu_pd(&chi[i]);
                            vclo = _mm256_loadu_pd(&clo[i]);
                            
                            //calc
                            _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &chi[i], vchi );
                            _mm256_storeu_pd( &clo[i], vclo );
                        }
                        sbhi = bhi[k];
                        sblo = blo[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,chi[i],clo[i]);
                            
                            //store
                            chi[i] = schi;
                            clo[i] = sclo;
                        }
                    }
                }
                break;
            }
            break;
    }
}
/**
 *
 * @fn _dd_mv_dd_ieee(double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-vector multiplication of double-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_mv_dd_ieee(double *chi, double *clo, double *ahi, double *alo, double *bhi, double *blo,int m, int n, int l, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int k;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        ch[k]=0;
                        cl[k]=0;
                    }
                    double sahi,salo,sbhi,sblo,schi,sclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k ++) {
                        //load
                        sbhi = bhi[k];
                        sblo = blo[k];
                        for (i = 0; i < m; i ++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd_ieee(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i = 0; i < m; i ++){
                        
                        _dd_add_dd_ieee(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double zhi[4] = {0,0,0,0};
                    double zlo[4] = {0,0,0,0};
                    double ch [num_threads * m];
                    double cl [num_threads * m];
                    for (k = 0; k < num_threads * m; k ++) {
                        ch[k] = 0;
                        cl[k] = 0;
                    }
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        //load
                        vbhi = _mm256_set_pd(bhi[k],0,0,0);
                        vblo = _mm256_set_pd(blo[k],0,0,0);
                        
                        for (i = 0; i < m; i++) {
                            //load
                            vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                            valo = _mm256_set_pd(alo[m*k+i],0,0,0);
                            vchi = _mm256_set_pd(ch[m*tid+i],0,0,0);
                            vclo = _mm256_set_pd(cl[m*tid+i],0,0,0);
                            
                            //calc
                            _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2_ieee(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &zhi[0], vchi );
                            _mm256_storeu_pd( &zlo[0], vclo );
                            
                            ch[m*tid+i]=zhi[3];
                            cl[m*tid+i]=zlo[3];
                        }
                    }
#pragma omp critical
                    for(i = 0; i < m; i ++){
                        
                        _dd_add_dd_ieee(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k, i, j;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    for (k = 0; k < num_threads * m; k++) {
                        ch[k] = 0;
                        cl[k] = 0;
                    }
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhi = _mm256_broadcast_sd(&bhi[k]);
                        vblo = _mm256_broadcast_sd(&blo[k]);
                        
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vchi = _mm256_loadu_pd(&ch[m*tid+i]);
                            vclo = _mm256_loadu_pd(&cl[m*tid+i]);
                            
                            //calc
                            _dd_mul_dd_avx2(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2_ieee(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &ch[m*tid+i], vchi );
                            _mm256_storeu_pd( &cl[m*tid+i], vclo );
                            
                        }
                        sbhi = bhi[k];
                        sblo = blo[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd_ieee(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        _dd_add_dd_ieee(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k, i, j;
                    int num_threads = omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double ch [num_threads*m];
                    double cl [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        ch[k]=0;
                        cl[k]=0;
                    }
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhi = _mm256_broadcast_sd(&bhi[k]);
                        vblo = _mm256_broadcast_sd(&blo[k]);
                        
                        for (i = 0; i < m-p; i+=4) {
                            
                            //load
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vchi = _mm256_loadu_pd(&ch[m*tid+i]);
                            vclo = _mm256_loadu_pd(&cl[m*tid+i]);
                            
                            //calc
                            _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2_ieee(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                            //store
                            _mm256_storeu_pd( &ch[m*tid+i], vchi );
                            _mm256_storeu_pd( &cl[m*tid+i], vclo );
                            
                        }
                        sbhi = bhi[k];
                        sblo = blo[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd_ieee(&schi,&sclo,schi,sclo,ch[m*tid+i],cl[m*tid+i]);
                            
                            //store
                            ch[m*tid+i] = schi;
                            cl[m*tid+i] = sclo;
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        _dd_add_dd_ieee(&schi,&sclo,chi[i],clo[i],ch[m*tid+i],cl[m*tid+i]);
                        
                        chi[i] = schi;
                        clo[i] = sclo;
                    }
                }
            }
            break;
    }
}

/**
 *
 * @fn _dd_tmv_dd(double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the transposed matrix-vector multiplication of double-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_tmv_dd( double *chi, double *clo, double *ahi,double *alo,double *bhi, double *blo,int m, int n, int l, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int k;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i;
                    double sahi,salo,sbhi,sblo,schi,sclo,ch,cl;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        ch=0;
                        cl=0;
                        for (i = 0; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            sbhi = bhi[i];
                            sblo = blo[i];
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd(&ch,&cl,schi,sclo,ch,cl);
                            
                        }
                        chi[k]=ch;
                        clo[k]=cl;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int k,i;
                    __m256d vahi,valo,vbhi,vblo,vchi,vclo,vzhi,vzlo;
                    double zhi[4]={0,0,0,0};
                    double zlo[4]={0,0,0,0};
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vchi = _mm256_set_pd(0,0,0,0);
                        vclo = _mm256_set_pd(0,0,0,0);
                        
                        for (i = 0; i < m; i++) {
                            //load
                            vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                            valo = _mm256_set_pd(alo[m*k+i],0,0,0);
                            vbhi = _mm256_set_pd(bhi[i],0,0,0);
                            vblo = _mm256_set_pd(blo[i],0,0,0);
                            
                            //calc
                            _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                        }
                        _mm256_storeu_pd( &zhi[0], vchi );
                        _mm256_storeu_pd( &zlo[0], vclo );
                        
                        chi[k]=zhi[3];
                        clo[k]=zlo[3];
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k;
                    double ch[4]={0,0,0,0};
                    double cl[4]={0,0,0,0};
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo,vch,vcl;
                    int i,j;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vch = _mm256_set_pd(0,0,0,0);
                        vcl = _mm256_set_pd(0,0,0,0);
                        
                        for (i = 0; i < m-p; i+=4) {
                            
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vbhi = _mm256_loadu_pd(&bhi[i]);
                            vblo = _mm256_loadu_pd(&blo[i]);
                            
                            //calc
                            _dd_mul_dd_avx2(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2(&vch,&vcl,vzhi,vzlo,vch,vcl);
                        }
                        _mm256_storeu_pd( &ch[0], vch );
                        _mm256_storeu_pd( &cl[0], vcl );
                        
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            sbhi = bhi[i];
                            sblo = blo[i];
                            
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[0],cl[0]);
                            
                            //store
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        
                        for(i=1;i<4;i++){
                            
                            _dd_add_dd(&schi,&sclo,ch[0],cl[0],ch[i],cl[i]);
                            
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        chi[k]=ch[0];
                        clo[k]=cl[0];
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k;
                    double ch[4]={0,0,0,0};
                    double cl[4]={0,0,0,0};
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo,vch,vcl;
                    int i,j;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vch = _mm256_set_pd(0,0,0,0);
                        vcl = _mm256_set_pd(0,0,0,0);
                        
                        for (i = 0; i < m-p; i+=4) {
                            
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vbhi = _mm256_loadu_pd(&bhi[i]);
                            vblo = _mm256_loadu_pd(&blo[i]);
                            
                            //calc
                            _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2(&vch,&vcl,vzhi,vzlo,vch,vcl);
                        }
                        _mm256_storeu_pd( &ch[0], vch );
                        _mm256_storeu_pd( &cl[0], vcl );
                        
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            sbhi = bhi[i];
                            sblo = blo[i];
                            
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd(&schi,&sclo,schi,sclo,ch[0],cl[0]);
                            
                            
                            //store
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        
                        for(i=1;i<4;i++){
                            
                            _dd_add_dd(&schi,&sclo,ch[0],cl[0],ch[i],cl[i]);
                            
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        chi[k]=ch[0];
                        clo[k]=cl[0];
                    }
                }
                break;
            }
            break;
    }
}

/**
 *
 * @fn _dd_tmv_dd_ieee(double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the transposed matrix-vector multiplication of double-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_tmv_dd_ieee( double *chi, double *clo, double *ahi,double *alo,double *bhi, double *blo,int m, int n, int l, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int k;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i;
                    double sahi,salo,sbhi,sblo,schi,sclo,ch,cl;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        ch=0;
                        cl=0;
                        for (i = 0; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            sbhi = bhi[i];
                            sblo = blo[i];
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd_ieee(&ch,&cl,schi,sclo,ch,cl);
                            
                        }
                        chi[k]=ch;
                        clo[k]=cl;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int k,i;
                    __m256d vahi,valo,vbhi,vblo,vchi,vclo,vzhi,vzlo;
                    double zhi[4]={0,0,0,0};
                    double zlo[4]={0,0,0,0};
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vchi = _mm256_set_pd(0,0,0,0);
                        vclo = _mm256_set_pd(0,0,0,0);
                        
                        for (i = 0; i < m; i++) {
                            //load
                            vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                            valo = _mm256_set_pd(alo[m*k+i],0,0,0);
                            vbhi = _mm256_set_pd(bhi[i],0,0,0);
                            vblo = _mm256_set_pd(blo[i],0,0,0);
                            
                            //calc
                            _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2_ieee(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                            
                        }
                        _mm256_storeu_pd( &zhi[0], vchi );
                        _mm256_storeu_pd( &zlo[0], vclo );
                        
                        chi[k]=zhi[3];
                        clo[k]=zlo[3];
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k;
                    double ch[4]={0,0,0,0};
                    double cl[4]={0,0,0,0};
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo,vch,vcl;
                    int i,j;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vch = _mm256_set_pd(0,0,0,0);
                        vcl = _mm256_set_pd(0,0,0,0);
                        
                        for (i = 0; i < m-p; i+=4) {
                            
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vbhi = _mm256_loadu_pd(&bhi[i]);
                            vblo = _mm256_loadu_pd(&blo[i]);
                            
                            //calc
                            _dd_mul_dd_avx2(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2_ieee(&vch,&vcl,vzhi,vzlo,vch,vcl);
                        }
                        _mm256_storeu_pd( &ch[0], vch );
                        _mm256_storeu_pd( &cl[0], vcl );
                        
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            sbhi = bhi[i];
                            sblo = blo[i];
                            
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd_ieee(&schi,&sclo,schi,sclo,ch[0],cl[0]);
                            
                            //store
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        
                        for(i=1;i<4;i++){
                            
                            _dd_add_dd_ieee(&schi,&sclo,ch[0],cl[0],ch[i],cl[i]);
                            
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        chi[k]=ch[0];
                        clo[k]=cl[0];
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m % 4;
                    int k;
                    double ch[4]={0,0,0,0};
                    double cl[4]={0,0,0,0};
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo,vch,vcl;
                    int i,j;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vch = _mm256_set_pd(0,0,0,0);
                        vcl = _mm256_set_pd(0,0,0,0);
                        
                        for (i = 0; i < m-p; i+=4) {
                            
                            vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                            valo = _mm256_loadu_pd(&alo[m*k+i]);
                            vbhi = _mm256_loadu_pd(&bhi[i]);
                            vblo = _mm256_loadu_pd(&blo[i]);
                            
                            //calc
                            _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                            _dd_add_dd_avx2_ieee(&vch,&vcl,vzhi,vzlo,vch,vcl);
                        }
                        _mm256_storeu_pd( &ch[0], vch );
                        _mm256_storeu_pd( &cl[0], vcl );
                        
                        for (i = m-p; i < m; i++) {
                            //load
                            sahi = ahi[m*k+i];
                            salo = alo[m*k+i];
                            sbhi = bhi[i];
                            sblo = blo[i];
                            
                            //calc
                            _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                            _dd_add_dd_ieee(&schi,&sclo,schi,sclo,ch[0],cl[0]);
                            
                            
                            //store
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        
                        for(i=1;i<4;i++){
                            
                            _dd_add_dd_ieee(&schi,&sclo,ch[0],cl[0],ch[i],cl[i]);
                            
                            ch[0] = schi;
                            cl[0] = sclo;
                        }
                        chi[k]=ch[0];
                        clo[k]=cl[0];
                    }
                }
                break;
            }
            break;
    }
}

/**
 *
 * @fn _dd_mm_dd(double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-matrix multiplication of double-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_mm_dd(double *chi, double *clo, double *ahi, double *alo,double *bhi, double *blo,int m, int n, int l, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int k;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    int i,j,k;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            for (i = 0; i < m; i++) {
                                //load
                                sahi = ahi[m*k+i];
                                salo = alo[m*k+i];
                                sbhi = bhi[l*j+k];
                                sblo = blo[l*j+k];
                                
                                //calc
                                _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                                _dd_add_dd(&schi,&sclo,schi,sclo,chi[m*j+i],clo[m*j+i]);
                                
                                //store
                                chi[m*j+i] = schi;
                                clo[m*j+i] = sclo;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahi,valo,vbhi,vblo,vchi,vclo,vzhi,vzlo;
                    double zhi[4]={0,0,0,0};
                    double zlo[4]={0,0,0,0};
                    int j,k,i;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhi = _mm256_set_pd(bhi[l*j+k],0,0,0);
                            vblo = _mm256_set_pd(blo[l*j+k],0,0,0);
                            for (i = 0; i < m; i++) {
                                vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                                valo = _mm256_set_pd(alo[m*k+i],0,0,0);
                                vchi = _mm256_set_pd(chi[m*j+i],0,0,0);
                                vclo = _mm256_set_pd(clo[m*j+i],0,0,0);
                                
                                //calc
                                _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                                _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                                
                                _mm256_storeu_pd( &zhi[0], vchi );
                                _mm256_storeu_pd( &zlo[0], vclo );
                                
                                chi[m*j+i]=zhi[3];
                                clo[m*j+i]=zlo[3];
                            }
                        }
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    int p=m%4;
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo,vch,vcl;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhi = _mm256_broadcast_sd(&bhi[l*j+k]);
                            vblo = _mm256_broadcast_sd(&blo[l*j+k]);
                            for (i = 0; i < m-p; i+=4) {
                                vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                                valo = _mm256_loadu_pd(&alo[m*k+i]);
                                vchi = _mm256_loadu_pd(&chi[m*j+i]);
                                vclo = _mm256_loadu_pd(&clo[m*j+i]);
                                
                                //calc
                                _dd_mul_dd_avx2(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                                _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                                
                                _mm256_storeu_pd( &chi[m*j+i], vchi );
                                _mm256_storeu_pd( &clo[m*j+i], vclo );
                            }
                            for (i = m-p; i < m; i++) {
                                //load
                                sahi = ahi[m*k+i];
                                salo = alo[m*k+i];
                                sbhi = bhi[l*j+k];
                                sblo = blo[l*j+k];
                                
                                //calc
                                _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                                _dd_add_dd(&schi,&sclo,schi,sclo,chi[m*j+i],clo[m*j+i]);
                                
                                //store
                                chi[m*j+i] = schi;
                                clo[m*j+i] = sclo;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    int p=m%4;
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo,vch,vcl;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhi = _mm256_broadcast_sd(&bhi[l*j+k]);
                            vblo = _mm256_broadcast_sd(&blo[l*j+k]);
                            for (i = 0; i < m-p; i+=4) {
                                vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                                valo = _mm256_loadu_pd(&alo[m*k+i]);
                                vchi = _mm256_loadu_pd(&chi[m*j+i]);
                                vclo = _mm256_loadu_pd(&clo[m*j+i]);
                                
                                //calc
                                _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                                _dd_add_dd_avx2(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                                
                                _mm256_storeu_pd( &chi[m*j+i], vchi );
                                _mm256_storeu_pd( &clo[m*j+i], vclo );
                            }
                            for (i = m-p; i < m; i++) {
                                //load
                                sahi = ahi[m*k+i];
                                salo = alo[m*k+i];
                                sbhi = bhi[l*j+k];
                                sblo = blo[l*j+k];
                                
                                //calc
                                _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                                _dd_add_dd(&schi,&sclo,schi,sclo,chi[m*j+i],clo[m*j+i]);
                                
                                //store
                                chi[m*j+i] = schi;
                                clo[m*j+i] = sclo;
                            }
                        }
                    }
                }
                break;
            }
            break;
    }
}

/**
 *
 * @fn _dd_mm_dd_ieee(double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-matrix multiplication of double-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_mm_dd_ieee(double *chi, double *clo, double *ahi, double *alo,double *bhi, double *blo,int m, int n, int l, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int k;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    int i,j,k;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            for (i = 0; i < m; i++) {
                                //load
                                sahi = ahi[m*k+i];
                                salo = alo[m*k+i];
                                sbhi = bhi[l*j+k];
                                sblo = blo[l*j+k];
                                
                                //calc
                                _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                                _dd_add_dd_ieee(&schi,&sclo,schi,sclo,chi[m*j+i],clo[m*j+i]);
                                
                                //store
                                chi[m*j+i] = schi;
                                clo[m*j+i] = sclo;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahi,valo,vbhi,vblo,vchi,vclo,vzhi,vzlo;
                    double zhi[4]={0,0,0,0};
                    double zlo[4]={0,0,0,0};
                    int j,k,i;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhi = _mm256_set_pd(bhi[l*j+k],0,0,0);
                            vblo = _mm256_set_pd(blo[l*j+k],0,0,0);
                            for (i = 0; i < m; i++) {
                                vahi = _mm256_set_pd(ahi[m*k+i],0,0,0);
                                valo = _mm256_set_pd(alo[m*k+i],0,0,0);
                                vchi = _mm256_set_pd(chi[m*j+i],0,0,0);
                                vclo = _mm256_set_pd(clo[m*j+i],0,0,0);
                                
                                //calc
                                _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                                _dd_add_dd_avx2_ieee(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                                
                                _mm256_storeu_pd( &zhi[0], vchi );
                                _mm256_storeu_pd( &zlo[0], vclo );
                                
                                chi[m*j+i]=zhi[3];
                                clo[m*j+i]=zlo[3];
                            }
                        }
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    int p=m%4;
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo,vch,vcl;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhi = _mm256_broadcast_sd(&bhi[l*j+k]);
                            vblo = _mm256_broadcast_sd(&blo[l*j+k]);
                            for (i = 0; i < m-p; i+=4) {
                                vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                                valo = _mm256_loadu_pd(&alo[m*k+i]);
                                vchi = _mm256_loadu_pd(&chi[m*j+i]);
                                vclo = _mm256_loadu_pd(&clo[m*j+i]);
                                
                                //calc
                                _dd_mul_dd_avx2(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                                _dd_add_dd_avx2_ieee(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                                
                                _mm256_storeu_pd( &chi[m*j+i], vchi );
                                _mm256_storeu_pd( &clo[m*j+i], vclo );
                            }
                            for (i = m-p; i < m; i++) {
                                //load
                                sahi = ahi[m*k+i];
                                salo = alo[m*k+i];
                                sbhi = bhi[l*j+k];
                                sblo = blo[l*j+k];
                                
                                //calc
                                _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                                _dd_add_dd_ieee(&schi,&sclo,schi,sclo,chi[m*j+i],clo[m*j+i]);
                                
                                //store
                                chi[m*j+i] = schi;
                                clo[m*j+i] = sclo;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    int p=m%4;
                    double sahi,salo,sbhi,sblo,schi,sclo;
                    __m256d vahi,valo,vbhi,vblo,vzhi,vzlo,vchi,vclo,vch,vcl;
#pragma omp for schedule(static)
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhi = _mm256_broadcast_sd(&bhi[l*j+k]);
                            vblo = _mm256_broadcast_sd(&blo[l*j+k]);
                            for (i = 0; i < m-p; i+=4) {
                                vahi = _mm256_loadu_pd(&ahi[m*k+i]);
                                valo = _mm256_loadu_pd(&alo[m*k+i]);
                                vchi = _mm256_loadu_pd(&chi[m*j+i]);
                                vclo = _mm256_loadu_pd(&clo[m*j+i]);
                                
                                //calc
                                _dd_mul_dd_fma(&vzhi,&vzlo,vahi,valo,vbhi,vblo);
                                _dd_add_dd_avx2_ieee(&vchi,&vclo,vzhi,vzlo,vchi,vclo);
                                
                                _mm256_storeu_pd( &chi[m*j+i], vchi );
                                _mm256_storeu_pd( &clo[m*j+i], vclo );
                            }
                            for (i = m-p; i < m; i++) {
                                //load
                                sahi = ahi[m*k+i];
                                salo = alo[m*k+i];
                                sbhi = bhi[l*j+k];
                                sblo = blo[l*j+k];
                                
                                //calc
                                _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
                                _dd_add_dd_ieee(&schi,&sclo,schi,sclo,chi[m*j+i],clo[m*j+i]);
                                
                                //store
                                chi[m*j+i] = schi;
                                clo[m*j+i] = sclo;
                            }
                        }
                    }
                }
                break;
            }
            break;
    }
}

//-----------------------------------------------------------------
// Double-Double - Quad-Double operation
//-----------------------------------------------------------------

/**
 *
 * @fn _dd_a_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int)
 * @brief You can compute the addition of double-double and quad-double numbers in a vector or matrix. You can choose the number of OpenMP threads and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl input (double-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _dd_a_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int omp_threadNum, int avx)
{
    int i;
    int p = m % 4;
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    switch(avx){
        case 0:
#pragma omp parallel num_threads(omp_threadNum)
        {
            double sahh,sahl,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
#pragma omp for
            for (i = 0; i < m * n; i ++){
                //load
                sahh = ahh[i];
                sahl = ahl[i];
                sbhh = bhh[i];
                sbhl = bhl[i];
                sblh = blh[i];
                sbll = bll[i];
                
                //calc
                _dd_add_qd(&schh,&schl,&sclh,&scll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                
                //store
                chh[i] = schh;
                chl[i] = schl;
                clh[i] = sclh;
                cll[i] = scll;
            }
        }
        break;
        case 1:
#pragma omp parallel num_threads(omp_threadNum)
        {
            double sahh,sahl,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
            __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll;
#pragma omp for
            for (i = 0; i < m * n - p; i += 4){
                //load
                vahh = _mm256_loadu_pd(&ahh[i]);
                vahl = _mm256_loadu_pd(&ahl[i]);
                vbhh = _mm256_loadu_pd(&bhh[i]);
                vbhl = _mm256_loadu_pd(&bhl[i]);
                vblh = _mm256_loadu_pd(&blh[i]);
                vbll = _mm256_loadu_pd(&bll[i]);
                
                //calc
                _dd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                
                //store
                _mm256_storeu_pd( &chh[i], vchh );
                _mm256_storeu_pd( &chl[i], vchl );
                _mm256_storeu_pd( &clh[i], vclh );
                _mm256_storeu_pd( &cll[i], vcll );
            }
#pragma omp for
            for (i = m * n - p; i < m * n; i ++){
                //load
                sahh = ahh[i];
                sahl = ahl[i];
                sbhh = bhh[i];
                sbhl = bhl[i];
                sblh = blh[i];
                sbll = bll[i];
                
                //calc
                _dd_add_qd(&schh,&schl,&sclh,&scll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                
                //store
                chh[i] = schh;
                chl[i] = schl;
                clh[i] = sclh;
                cll[i] = scll;
            }
        }
        break;
        
    }
}
/**
 *
 * @fn _dd_scl_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the scaler of double-double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl input (double-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _dd_scl_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sahl,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
                    sahh = ahh[0];
                    sahl = ahl[0];
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _dd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll;
                    double vzhh[4]={0,0,0,0};
                    double vzhl[4]={0,0,0,0};
                    double vzlh[4]={0,0,0,0};
                    double vzll[4]={0,0,0,0};
                    vahh = _mm256_set_pd(ahh[0],0,0,0);
                    vahl = _mm256_set_pd(ahl[0],0,0,0);
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vbhh = _mm256_set_pd(bhh[i],0,0,0);
                        vbhl = _mm256_set_pd(bhl[i],0,0,0);
                        vblh = _mm256_set_pd(blh[i],0,0,0);
                        vbll = _mm256_set_pd(bll[i],0,0,0);
                        
                        //calc
                        _dd_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                        
                        //store
                        _mm256_storeu_pd( &vzhh[0], vchh );
                        _mm256_storeu_pd( &vzhl[0], vchl );
                        _mm256_storeu_pd( &vzlh[0], vclh );
                        _mm256_storeu_pd( &vzll[0], vcll );
                        
                        chh[i] = vzhh[3];
                        chl[i] = vzhl[3];
                        clh[i] = vzlh[3];
                        cll[i] = vzll[3];
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sahl,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll;
                    sahh = ahh[0];
                    sahl = ahl[0];
                    vahh = _mm256_broadcast_sd(&ahh[0]);
                    vahl = _mm256_broadcast_sd(&ahl[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _dd_mul_qd_avx2(&vchh,&vchl,&vclh,&vcll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                                                
                        //store
                        _mm256_storeu_pd( &chh[i], vchh );
                        _mm256_storeu_pd( &chl[i], vchl );
                        _mm256_storeu_pd( &clh[i], vclh );
                        _mm256_storeu_pd( &cll[i], vcll );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _dd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sahl,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll;
                    sahh = ahh[0];
                    sahl = ahl[0];
                    vahh = _mm256_broadcast_sd(&ahh[0]);
                    vahl = _mm256_broadcast_sd(&ahl[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _dd_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                                                
                        //store
                        _mm256_storeu_pd( &chh[i], vchh );
                        _mm256_storeu_pd( &chl[i], vchl );
                        _mm256_storeu_pd( &clh[i], vclh );
                        _mm256_storeu_pd( &cll[i], vcll );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _dd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
            }
    }
}
/**
 *
 * @fn _dd_dot_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the inner product of double-double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl input (double-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _dd_dot_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sahl,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads];
                    double tchl [num_threads];
                    double tclh [num_threads];
                    double tcll [num_threads];
                    for (i = 0; i < num_threads; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sahh = ahh[i];
                        sahl = ahl[i];
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _dd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                        _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[tid],tchl[tid],tclh[tid],tcll[tid],schh,schl,sclh,scll);
                        
                        //store
                        tchh[tid] = szhh;
                        tchl[tid] = szhl;
                        tclh[tid] = szlh;
                        tcll[tid] = szll;
                    }
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[tid],tchl[tid],tclh[tid],tcll[tid],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads*4];
                    double tchl [num_threads*4];
                    double tclh [num_threads*4];
                    double tcll [num_threads*4];
                    for (i = 0; i < num_threads*4; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vzhh,vzhl,vzlh,vzll;
                    __m256d vchh = _mm256_setzero_pd();
                    __m256d vchl = _mm256_setzero_pd();
                    __m256d vclh = _mm256_setzero_pd();
                    __m256d vcll = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vahh = _mm256_set_pd(ahh[i],0,0,0);
                        vahl = _mm256_set_pd(ahl[i],0,0,0);
                        vbhh = _mm256_set_pd(bhh[i],0,0,0);
                        vbhl = _mm256_set_pd(bhl[i],0,0,0);
                        vblh = _mm256_set_pd(blh[i],0,0,0);
                        vbll = _mm256_set_pd(bll[i],0,0,0);
                        
                        //calc
                        _dd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        
                    }
                    //store
                    _mm256_storeu_pd( &tchh[4*tid], vchh );
                    _mm256_storeu_pd( &tchl[4*tid], vchl );
                    _mm256_storeu_pd( &tclh[4*tid], vclh );
                    _mm256_storeu_pd( &tcll[4*tid], vcll );
                    
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid+3],tchl[4*tid+3],tclh[4*tid+3],tcll[4*tid+3],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sahl,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads*4];
                    double tchl [num_threads*4];
                    double tclh [num_threads*4];
                    double tcll [num_threads*4];
                    for (i = 0; i < num_threads*4; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vzhh,vzhl,vzlh,vzll;
                    __m256d vchh = _mm256_setzero_pd();
                    __m256d vchl = _mm256_setzero_pd();
                    __m256d vclh = _mm256_setzero_pd();
                    __m256d vcll = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahh = _mm256_loadu_pd(&ahh[i]);
                        vahl = _mm256_loadu_pd(&ahl[i]);
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _dd_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        
                    }
                    //store
                    _mm256_storeu_pd( &tchh[4*tid], vchh );
                    _mm256_storeu_pd( &tchl[4*tid], vchl );
                    _mm256_storeu_pd( &tclh[4*tid], vclh );
                    _mm256_storeu_pd( &tcll[4*tid], vcll );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        
                        //load
                        sahh = ahh[i];
                        sahl = ahl[i];
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _dd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                        _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],schh,schl,sclh,scll);
                        
                        //store
                        tchh[4*tid] = szhh;
                        tchl[4*tid] = szhl;
                        tclh[4*tid] = szlh;
                        tcll[4*tid] = szll;
                        
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid+i],tchl[4*tid+i],tclh[4*tid+i],tcll[4*tid+i],tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid]);
                        
                        //store
                        tchh[4*tid] = schh;
                        tchl[4*tid] = schl;
                        tclh[4*tid] = sclh;
                        tcll[4*tid] = scll;
                    }
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sahl,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads*4];
                    double tchl [num_threads*4];
                    double tclh [num_threads*4];
                    double tcll [num_threads*4];
                    for (i = 0; i < num_threads*4; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vzhh,vzhl,vzlh,vzll;
                    __m256d vchh = _mm256_setzero_pd();
                    __m256d vchl = _mm256_setzero_pd();
                    __m256d vclh = _mm256_setzero_pd();
                    __m256d vcll = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahh = _mm256_loadu_pd(&ahh[i]);
                        vahl = _mm256_loadu_pd(&ahl[i]);
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _dd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        
                    }
                    //store
                    _mm256_storeu_pd( &tchh[4*tid], vchh );
                    _mm256_storeu_pd( &tchl[4*tid], vchl );
                    _mm256_storeu_pd( &tclh[4*tid], vclh );
                    _mm256_storeu_pd( &tcll[4*tid], vcll );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        
                        //load
                        sahh = ahh[i];
                        sahl = ahl[i];
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _dd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                        _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],schh,schl,sclh,scll);
                        
                        //store
                        tchh[4*tid] = szhh;
                        tchl[4*tid] = szhl;
                        tclh[4*tid] = szlh;
                        tcll[4*tid] = szll;
                        
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid+i],tchl[4*tid+i],tclh[4*tid+i],tcll[4*tid+i],tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid]);
                        
                        //store
                        tchh[4*tid] = schh;
                        tchl[4*tid] = schl;
                        tclh[4*tid] = sclh;
                        tcll[4*tid] = scll;
                    }
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                    }
                }
                break;
            }
    }
}
/**
 *
 * @fn _dd_mv_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-vector multiplicaion of double-double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl input (double-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _dd_mv_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
                    
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sahl,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        sbhh = bhh[k];
                        sbhl = bhl[k];
                        sblh = blh[k];
                        sbll = bll[k];
                        for (i = 0; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            //calc
                            _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                            
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int k,i;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double vte1[4]={0,0,0,0};
                    double vte2[4]={0,0,0,0};
                    double vte3[4]={0,0,0,0};
                    double vte4[4]={0,0,0,0};
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double schh,schl,sclh,scll;
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_set_pd(bhh[k],0,0,0);
                        vbhl = _mm256_set_pd(bhl[k],0,0,0);
                        vblh = _mm256_set_pd(blh[k],0,0,0);
                        vbll = _mm256_set_pd(bll[k],0,0,0);
                        for (i = 0; i < m; i++) {
                            //load
                            vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                            vahl = _mm256_set_pd(ahl[m*k+i],0,0,0);
                            vzhh = _mm256_set_pd(tchh[m*tid+i],0,0,0);
                            vzhl = _mm256_set_pd(tchl[m*tid+i],0,0,0);
                            vzlh = _mm256_set_pd(tclh[m*tid+i],0,0,0);
                            vzll = _mm256_set_pd(tcll[m*tid+i],0,0,0);
                            
                            //calc
                            _dd_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            //store
                            _mm256_storeu_pd( &vte1[0], vchh );
                            _mm256_storeu_pd( &vte2[0], vchl );
                            _mm256_storeu_pd( &vte3[0], vclh );
                            _mm256_storeu_pd( &vte4[0], vcll );
                            
                            tchh[m*tid+i]=vte1[3];
                            tchl[m*tid+i]=vte2[3];
                            tclh[m*tid+i]=vte3[3];
                            tcll[m*tid+i]=vte4[3];
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sahl,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_broadcast_sd(&bhh[k]);
                        vbhl = _mm256_broadcast_sd(&bhl[k]);
                        vblh = _mm256_broadcast_sd(&blh[k]);
                        vbll = _mm256_broadcast_sd(&bll[k]);
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            vzhh = _mm256_loadu_pd(&tchh[m*tid+i]);
                            vzhl = _mm256_loadu_pd(&tchl[m*tid+i]);
                            vzlh = _mm256_loadu_pd(&tclh[m*tid+i]);
                            vzll = _mm256_loadu_pd(&tcll[m*tid+i]);
                            
                            //calc
                            _dd_mul_qd_avx2(&vchh,&vchl,&vclh,&vcll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            
                            //store
                            _mm256_storeu_pd( &tchh[m*tid+i], vchh );
                            _mm256_storeu_pd( &tchl[m*tid+i], vchl );
                            _mm256_storeu_pd( &tclh[m*tid+i], vclh );
                            _mm256_storeu_pd( &tcll[m*tid+i], vcll );
                        }
                        sbhh = bhh[k];
                        sbhl = bhl[k];
                        sblh = blh[k];
                        sbll = bll[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            
                            //calc
                            _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sahl,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_broadcast_sd(&bhh[k]);
                        vbhl = _mm256_broadcast_sd(&bhl[k]);
                        vblh = _mm256_broadcast_sd(&blh[k]);
                        vbll = _mm256_broadcast_sd(&bll[k]);
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            vzhh = _mm256_loadu_pd(&tchh[m*tid+i]);
                            vzhl = _mm256_loadu_pd(&tchl[m*tid+i]);
                            vzlh = _mm256_loadu_pd(&tclh[m*tid+i]);
                            vzll = _mm256_loadu_pd(&tcll[m*tid+i]);
                            
                            //calc
                            _dd_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            
                            //store
                            _mm256_storeu_pd( &tchh[m*tid+i], vchh );
                            _mm256_storeu_pd( &tchl[m*tid+i], vchl );
                            _mm256_storeu_pd( &tclh[m*tid+i], vclh );
                            _mm256_storeu_pd( &tcll[m*tid+i], vcll );
                        }
                        sbhh = bhh[k];
                        sbhl = bhl[k];
                        sblh = blh[k];
                        sbll = bll[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            
                            //calc
                            _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                    
                }
                
                break;
            }
            break;
    }
}
/**
 *
 * @fn _dd_tmv_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the transposed matrix-vector multiplicaion of double-double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl input (double-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _dd_tmv_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    double schh,schl,sclh,scll,sahh,sahl,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        schh=0;
                        schl=0;
                        sclh=0;
                        scll=0;
                        for (i = 0; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            sbhh = bhh[i];
                            sbhl = bhl[i];
                            sblh = blh[i];
                            sbll = bll[i];
                            
                            //calc
                            _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,schh,schl,sclh,scll);
                            
                            
                            //store
                            schh = szhh;
                            schl = szhl;
                            sclh = szlh;
                            scll = szll;
                        }
                        chh[k] = schh;
                        chl[k] = schl;
                        clh[k] = sclh;
                        cll[k] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m; i++) {
                            //load
                            vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                            vahl = _mm256_set_pd(ahl[m*k+i],0,0,0);
                            vbhh = _mm256_set_pd(bhh[i],0,0,0);
                            vbhl = _mm256_set_pd(bhl[i],0,0,0);
                            vblh = _mm256_set_pd(blh[i],0,0,0);
                            vbll = _mm256_set_pd(bll[i],0,0,0);
                            
                            //calc
                            _dd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        
                        chh[k] = tchh[3];
                        chl[k] = tchl[3];
                        clh[k] = tclh[3];
                        cll[k] = tcll[3];
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    double schh,schl,sclh,scll,sahh,sahl,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            vbhh = _mm256_loadu_pd(&bhh[i]);
                            vbhl = _mm256_loadu_pd(&bhl[i]);
                            vblh = _mm256_loadu_pd(&blh[i]);
                            vbll = _mm256_loadu_pd(&bll[i]);
                            
                            //calc
                            _dd_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            sbhh = bhh[i];
                            sbhl = bhl[i];
                            sblh = blh[i];
                            sbll = bll[i];
                            
                            //calc
                            _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        for(i=1;i<4;i++){
                            //calc
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[i],tchl[i],tclh[i],tcll[i],tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        chh[k] = tchh[0];
                        chl[k] = tchl[0];
                        clh[k] = tclh[0];
                        cll[k] = tcll[0];
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    double schh,schl,sclh,scll,sahh,sahl,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            vbhh = _mm256_loadu_pd(&bhh[i]);
                            vbhl = _mm256_loadu_pd(&bhl[i]);
                            vblh = _mm256_loadu_pd(&blh[i]);
                            vbll = _mm256_loadu_pd(&bll[i]);
                            
                            //calc
                            _dd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            sbhh = bhh[i];
                            sbhl = bhl[i];
                            sblh = blh[i];
                            sbll = bll[i];
                            
                            //calc
                            _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        for(i=1;i<4;i++){
                            //calc
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[i],tchl[i],tclh[i],tcll[i],tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        chh[k] = tchh[0];
                        chl[k] = tchl[0];
                        clh[k] = tclh[0];
                        cll[k] = tcll[0];
                    }
                }
                break;
            }
            break;
    }
}
/**
 *
 * @fn _dd_mm_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-matrix multiplicaion of double-double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl input (double-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _dd_mm_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    double sahh,sahl,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            sbhh = bhh[l*j+k];
                            sbhl = bhl[l*j+k];
                            sblh = blh[l*j+k];
                            sbll = bll[l*j+k];
                            for (i = 0; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                sahl = ahl[m*k+i];
                                
                                //calc
                                _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
                    double szhh[4]={0,0,0,0};
                    double szhl[4]={0,0,0,0};
                    double szlh[4]={0,0,0,0};
                    double szll[4]={0,0,0,0};
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_set_pd(bhh[l*j+k],0,0,0);
                            vbhl = _mm256_set_pd(bhl[l*j+k],0,0,0);
                            vblh = _mm256_set_pd(blh[l*j+k],0,0,0);
                            vbll = _mm256_set_pd(bll[l*j+k],0,0,0);
                            
                            for (i = 0; i < m; i++) {
                                vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                                vahl = _mm256_set_pd(ahl[m*k+i],0,0,0);
                                vchh = _mm256_set_pd(chh[m*j+i],0,0,0);
                                vchl = _mm256_set_pd(chl[m*j+i],0,0,0);
                                vclh = _mm256_set_pd(clh[m*j+i],0,0,0);
                                vcll = _mm256_set_pd(cll[m*j+i],0,0,0);
                                
                                //calc
                                _dd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                _mm256_storeu_pd( &szhh[0], vchh );
                                _mm256_storeu_pd( &szhl[0], vchl );
                                _mm256_storeu_pd( &szlh[0], vclh );
                                _mm256_storeu_pd( &szll[0], vcll );
                                
                                chh[m*j+i] = szhh[3];
                                chl[m*j+i] = szhl[3];
                                clh[m*j+i] = szlh[3];
                                cll[m*j+i] = szll[3];
                            }
                        }
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    int p=m%4;
                    double sahh,sahl,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_broadcast_sd(&bhh[l*j+k]);
                            vbhl = _mm256_broadcast_sd(&bhl[l*j+k]);
                            vblh = _mm256_broadcast_sd(&blh[l*j+k]);
                            vbll = _mm256_broadcast_sd(&bll[l*j+k]);
                            
                            for (i = 0; i < m-p; i+=4) {
                                //load
                                vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                                vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                                vchh = _mm256_loadu_pd(&chh[m*j+i]);
                                vchl = _mm256_loadu_pd(&chl[m*j+i]);
                                vclh = _mm256_loadu_pd(&clh[m*j+i]);
                                vcll = _mm256_loadu_pd(&cll[m*j+i]);
                                
                                //calc
                                _dd_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                //store
                                _mm256_storeu_pd( &chh[m*j+i], vchh );
                                _mm256_storeu_pd( &chl[m*j+i], vchl );
                                _mm256_storeu_pd( &clh[m*j+i], vclh );
                                _mm256_storeu_pd( &cll[m*j+i], vcll );
                            }
                            sbhh = bhh[l*j+k];
                            sbhl = bhl[l*j+k];
                            sblh = blh[l*j+k];
                            sbll = bll[l*j+k];
                            for (i = m-p; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                sahl = ahl[m*k+i];
                                
                                //calc
                                _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    int p=m%4;
                    double sahh,sahl,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_broadcast_sd(&bhh[l*j+k]);
                            vbhl = _mm256_broadcast_sd(&bhl[l*j+k]);
                            vblh = _mm256_broadcast_sd(&blh[l*j+k]);
                            vbll = _mm256_broadcast_sd(&bll[l*j+k]);
                            
                            for (i = 0; i < m-p; i+=4) {
                                //load
                                vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                                vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                                vchh = _mm256_loadu_pd(&chh[m*j+i]);
                                vchl = _mm256_loadu_pd(&chl[m*j+i]);
                                vclh = _mm256_loadu_pd(&clh[m*j+i]);
                                vcll = _mm256_loadu_pd(&cll[m*j+i]);
                                
                                //calc
                                _dd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,vbhh,vbhl,vblh,vbll);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                //store
                                _mm256_storeu_pd( &chh[m*j+i], vchh );
                                _mm256_storeu_pd( &chl[m*j+i], vchl );
                                _mm256_storeu_pd( &clh[m*j+i], vclh );
                                _mm256_storeu_pd( &cll[m*j+i], vcll );
                            }
                            sbhh = bhh[l*j+k];
                            sbhl = bhl[l*j+k];
                            sblh = blh[l*j+k];
                            sbll = bll[l*j+k];
                            for (i = m-p; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                sahl = ahl[m*k+i];
                                
                                //calc
                                _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,sbhh,sbhl,sblh,sbll);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
            }
            break;
    }
}
//-----------------------------------------------------------------
// Quad-Double - Double operation
//-----------------------------------------------------------------

/**
 *
 * @fn _qd_scl_d(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the scaler of quad-double and double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh input (double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _qd_scl_d(double* chh,double* chl,double* clh,double* cll,double *ahh, double *ahl,double *alh,double *all, double *bhh,int m, int n, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sahl,salh,sall,sbhh,schh,schl,sclh,scll;
                    sahh = ahh[0];
                    sahl = ahl[0];
                    salh = alh[0];
                    sall = all[0];
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        
                        //calc
                        _d_mul_qd(&schh,&schl,&sclh,&scll,sbhh,sahh,sahl,salh,sall);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahh,vahl,valh,vall,vbhh,vchh,vchl,vclh,vcll;
                    double vzhh[4]={0,0,0,0};
                    double vzhl[4]={0,0,0,0};
                    double vzlh[4]={0,0,0,0};
                    double vzll[4]={0,0,0,0};
                    vahh = _mm256_set_pd(ahh[0],0,0,0);
                    vahl = _mm256_set_pd(ahl[0],0,0,0);
                    valh = _mm256_set_pd(alh[0],0,0,0);
                    vall = _mm256_set_pd(all[0],0,0,0);
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vbhh = _mm256_set_pd(bhh[i],0,0,0);
                        
                        //calc
                        _d_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vbhh,vahh,vahl,valh,vall);
                        
                        //store
                        _mm256_storeu_pd( &vzhh[0], vchh );
                        _mm256_storeu_pd( &vzhl[0], vchl );
                        _mm256_storeu_pd( &vzlh[0], vclh );
                        _mm256_storeu_pd( &vzll[0], vcll );
                        
                        chh[i] = vzhh[3];
                        chl[i] = vzhl[3];
                        clh[i] = vzlh[3];
                        cll[i] = vzll[3];
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sahl,salh,sall,sbhh,schh,schl,sclh,scll;
                    __m256d vahh,vahl,valh,vall,vbhh,vchh,vchl,vclh,vcll;
                    sahh = ahh[0];
                    sahl = ahl[0];
                    salh = alh[0];
                    sall = all[0];
                    vahh = _mm256_broadcast_sd(&ahh[0]);
                    vahl = _mm256_broadcast_sd(&ahl[0]);
                    valh = _mm256_broadcast_sd(&alh[0]);
                    vall = _mm256_broadcast_sd(&all[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        
                        //calc
                        _d_mul_qd_avx2(&vchh,&vchl,&vclh,&vcll,vbhh,vahh,vahl,valh,vall);
                        
                        //store
                        _mm256_storeu_pd( &chh[i], vchh );
                        _mm256_storeu_pd( &chl[i], vchl );
                        _mm256_storeu_pd( &clh[i], vclh );
                        _mm256_storeu_pd( &cll[i], vcll );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        
                        //calc
                        _d_mul_qd(&schh,&schl,&sclh,&scll,sbhh,sahh,sahl,salh,sall);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sbhh,sahl,salh,sall,schh,schl,sclh,scll;
                    __m256d vahh,vbhh,vahl,valh,vall,vchh,vchl,vclh,vcll;
                    sahh = ahh[0];
                    sahl = ahl[0];
                    salh = alh[0];
                    sall = all[0];
                    vahh = _mm256_broadcast_sd(&ahh[0]);
                    vahl = _mm256_broadcast_sd(&ahl[0]);
                    valh = _mm256_broadcast_sd(&alh[0]);
                    vall = _mm256_broadcast_sd(&all[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        
                        //calc
                        _d_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vbhh,vahh,vahl,valh,vall);
                        
                        //store
                        _mm256_storeu_pd( &chh[i], vchh );
                        _mm256_storeu_pd( &chl[i], vchl );
                        _mm256_storeu_pd( &clh[i], vclh );
                        _mm256_storeu_pd( &cll[i], vcll );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        
                        //calc
                        _d_mul_qd(&schh,&schl,&sclh,&scll,sbhh,sahh,sahl,salh,sall);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
            }
    }
}
/**
 *
 * @fn _qd_mv_d(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-vector multiplicaion of quad-double and double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh input (double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _qd_mv_d(double* chh,double* chl,double* clh,double* cll,double *bhh,double *ahh,double *ahl,double *alh,double *all, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
                    
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sbhh,sahl,salh,sall,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        sbhh = bhh[k];
                        for (i = 0; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            
                            //calc
                            _d_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sahh,sahl,salh,sall);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                            
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int k,i;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double vte1[4]={0,0,0,0};
                    double vte2[4]={0,0,0,0};
                    double vte3[4]={0,0,0,0};
                    double vte4[4]={0,0,0,0};
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double schh,schl,sclh,scll;
                    __m256d vahh,vbhh,vahl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_set_pd(bhh[k],0,0,0);
                        for (i = 0; i < m; i++) {
                            //load
                            vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                            vahl = _mm256_set_pd(ahl[m*k+i],0,0,0);
                            valh = _mm256_set_pd(alh[m*k+i],0,0,0);
                            vall = _mm256_set_pd(all[m*k+i],0,0,0);
                            vzhh = _mm256_set_pd(tchh[m*tid+i],0,0,0);
                            vzhl = _mm256_set_pd(tchl[m*tid+i],0,0,0);
                            vzlh = _mm256_set_pd(tclh[m*tid+i],0,0,0);
                            vzll = _mm256_set_pd(tcll[m*tid+i],0,0,0);
                            
                            //calc
                            _d_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vbhh,vahh,vahl,valh,vall);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            //store
                            _mm256_storeu_pd( &vte1[0], vchh );
                            _mm256_storeu_pd( &vte2[0], vchl );
                            _mm256_storeu_pd( &vte3[0], vclh );
                            _mm256_storeu_pd( &vte4[0], vcll );
                            
                            tchh[m*tid+i]=vte1[3];
                            tchl[m*tid+i]=vte2[3];
                            tclh[m*tid+i]=vte3[3];
                            tcll[m*tid+i]=vte4[3];
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sbhh,sahl,salh,sall,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vbhh,vahl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_broadcast_sd(&bhh[k]);
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            valh = _mm256_loadu_pd(&alh[m*k+i]);
                            vall = _mm256_loadu_pd(&all[m*k+i]);
                            vzhh = _mm256_loadu_pd(&tchh[m*tid+i]);
                            vzhl = _mm256_loadu_pd(&tchl[m*tid+i]);
                            vzlh = _mm256_loadu_pd(&tclh[m*tid+i]);
                            vzll = _mm256_loadu_pd(&tcll[m*tid+i]);
                            
                            //calc
                            _d_mul_qd_avx2(&vchh,&vchl,&vclh,&vcll,vbhh,vahh,vahl,valh,vall);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            //store
                            _mm256_storeu_pd( &tchh[m*tid+i], vchh );
                            _mm256_storeu_pd( &tchl[m*tid+i], vchl );
                            _mm256_storeu_pd( &tclh[m*tid+i], vclh );
                            _mm256_storeu_pd( &tcll[m*tid+i], vcll );
                        }
                        sbhh = bhh[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            
                            //calc
                            _d_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sahh,sahl,salh,sall);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sbhh,sahl,salh,sall,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vbhh,vahl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_broadcast_sd(&bhh[k]);
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            valh = _mm256_loadu_pd(&alh[m*k+i]);
                            vall = _mm256_loadu_pd(&all[m*k+i]);
                            vzhh = _mm256_loadu_pd(&tchh[m*tid+i]);
                            vzhl = _mm256_loadu_pd(&tchl[m*tid+i]);
                            vzlh = _mm256_loadu_pd(&tclh[m*tid+i]);
                            vzll = _mm256_loadu_pd(&tcll[m*tid+i]);
                            
                            //calc
                            _d_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vbhh,vahh,vahl,valh,vall);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            //store
                            _mm256_storeu_pd( &tchh[m*tid+i], vchh );
                            _mm256_storeu_pd( &tchl[m*tid+i], vchl );
                            _mm256_storeu_pd( &tclh[m*tid+i], vclh );
                            _mm256_storeu_pd( &tcll[m*tid+i], vcll );
                        }
                        sbhh = bhh[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            
                            //calc
                            _d_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sahh,sahl,salh,sall);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                    
                }
                
                break;
            }
            break;
    }
}
/**
 *
 * @fn _qd_tmv_d(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the transposed matrix-vector multiplicaion of quad-double and double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh input (double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _qd_tmv_d(double* chh,double* chl,double* clh,double* cll,double *bhh,double *ahh,double *ahl,double *alh,double *all, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    double schh,schl,sclh,scll,sahh,sbhh,sahl,salh,sall,szhh,szhl,szlh,szll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        schh=0;
                        schl=0;
                        sclh=0;
                        scll=0;
                        for (i = 0; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            sbhh = bhh[i];
                            
                            //calc
                            _d_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sahh,sahl,salh,sall);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,schh,schl,sclh,scll);
                            
                            
                            //store
                            schh = szhh;
                            schl = szhl;
                            sclh = szlh;
                            scll = szll;
                        }
                        chh[k] = schh;
                        chl[k] = schl;
                        clh[k] = sclh;
                        cll[k] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    __m256d vahh,vbhh,vahl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m; i++) {
                            //load
                            vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                            vahl = _mm256_set_pd(ahl[m*k+i],0,0,0);
                            valh = _mm256_set_pd(alh[m*k+i],0,0,0);
                            vall = _mm256_set_pd(all[m*k+i],0,0,0);
                            vbhh = _mm256_set_pd(bhh[i],0,0,0);
                            
                            //calc
                            _d_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vbhh,vahh,vahl,valh,vall);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        
                        chh[k] = tchh[3];
                        chl[k] = tchl[3];
                        clh[k] = tclh[3];
                        cll[k] = tcll[3];
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    double schh,schl,sclh,scll,sahh,sbhh,sahl,salh,sall,szhh,szhl,szlh,szll;
                    __m256d vahh,vbhh,vahl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            valh = _mm256_loadu_pd(&alh[m*k+i]);
                            vall = _mm256_loadu_pd(&all[m*k+i]);
                            vbhh = _mm256_loadu_pd(&bhh[i]);
                            
                            //calc
                            _d_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vbhh,vahh,vahl,valh,vall);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            sbhh = bhh[i];
                            
                            //calc
                            _d_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sahh,sahl,salh,sall);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        for(i=1;i<4;i++){
                            //calc
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[i],tchl[i],tclh[i],tcll[i],tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        chh[k] = tchh[0];
                        chl[k] = tchl[0];
                        clh[k] = tclh[0];
                        cll[k] = tcll[0];
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    double schh,schl,sclh,scll,sahh,sbhh,sahl,salh,sall,szhh,szhl,szlh,szll;
                    __m256d vahh,vbhh,vahl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            valh = _mm256_loadu_pd(&alh[m*k+i]);
                            vall = _mm256_loadu_pd(&all[m*k+i]);
                            vbhh = _mm256_loadu_pd(&bhh[i]);
                            
                            //calc
                            _d_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vbhh,vahh,vahl,valh,vall);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            sbhh = bhh[i];
                            
                            //calc
                            _d_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sahh,sahl,salh,sall);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        for(i=1;i<4;i++){
                            //calc
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[i],tchl[i],tclh[i],tcll[i],tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        chh[k] = tchh[0];
                        chl[k] = tchl[0];
                        clh[k] = tclh[0];
                        cll[k] = tcll[0];
                    }
                }
                break;
            }
            break;
    }
}
/**
 *
 * @fn _qd_mm_d(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-matrix multiplicaion of quad-double and double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh input (double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _qd_mm_d(double* chh,double* chl,double* clh,double* cll,double *bhh,double *ahh,double *ahl,double *alh,double *all, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    double sahh,sbhh,sahl,salh,sall,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            sbhh = bhh[l*j+k];
                            for (i = 0; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                sahl = ahl[m*k+i];
                                salh = alh[m*k+i];
                                sall = all[m*k+i];
                                
                                //calc
                                _d_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sahh,sahl,salh,sall);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    __m256d vahh,vbhh,vahl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
                    double szhh[4]={0,0,0,0};
                    double szhl[4]={0,0,0,0};
                    double szlh[4]={0,0,0,0};
                    double szll[4]={0,0,0,0};
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_set_pd(bhh[l*j+k],0,0,0);
                            
                            for (i = 0; i < m; i++) {
                                vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                                vahl = _mm256_set_pd(ahl[m*k+i],0,0,0);
                                valh = _mm256_set_pd(alh[m*k+i],0,0,0);
                                vall = _mm256_set_pd(all[m*k+i],0,0,0);
                                
                                vchh = _mm256_set_pd(chh[m*j+i],0,0,0);
                                vchl = _mm256_set_pd(chl[m*j+i],0,0,0);
                                vclh = _mm256_set_pd(clh[m*j+i],0,0,0);
                                vcll = _mm256_set_pd(cll[m*j+i],0,0,0);
                                
                                //calc
                                _d_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vbhh,vahh,vahl,valh,vall);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                _mm256_storeu_pd( &szhh[0], vchh );
                                _mm256_storeu_pd( &szhl[0], vchl );
                                _mm256_storeu_pd( &szlh[0], vclh );
                                _mm256_storeu_pd( &szll[0], vcll );
                                
                                chh[m*j+i] = szhh[3];
                                chl[m*j+i] = szhl[3];
                                clh[m*j+i] = szlh[3];
                                cll[m*j+i] = szll[3];
                            }
                        }
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    int p=m%4;
                    double sahh,sbhh,sahl,salh,sall,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vbhh,vahl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_broadcast_sd(&bhh[l*j+k]);
                            
                            for (i = 0; i < m-p; i+=4) {
                                //load
                                vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                                vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                                valh = _mm256_loadu_pd(&alh[m*k+i]);
                                vall = _mm256_loadu_pd(&all[m*k+i]);
                                vchh = _mm256_loadu_pd(&chh[m*j+i]);
                                vchl = _mm256_loadu_pd(&chl[m*j+i]);
                                vclh = _mm256_loadu_pd(&clh[m*j+i]);
                                vcll = _mm256_loadu_pd(&cll[m*j+i]);
                                
                                //calc
                                _d_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vbhh,vahh,vahl,valh,vall);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                //store
                                _mm256_storeu_pd( &chh[m*j+i], vchh );
                                _mm256_storeu_pd( &chl[m*j+i], vchl );
                                _mm256_storeu_pd( &clh[m*j+i], vclh );
                                _mm256_storeu_pd( &cll[m*j+i], vcll );
                            }
                            sbhh = bhh[l*j+k];
                            for (i = m-p; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                sahl = ahl[m*k+i];
                                salh = alh[m*k+i];
                                sall = all[m*k+i];
                                
                                //calc
                                _d_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sahh,sahl,salh,sall);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    int p=m%4;
                    double sahh,sbhh,sahl,salh,sall,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vbhh,vahl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_broadcast_sd(&bhh[l*j+k]);
                            
                            for (i = 0; i < m-p; i+=4) {
                                //load
                                vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                                vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                                valh = _mm256_loadu_pd(&alh[m*k+i]);
                                vall = _mm256_loadu_pd(&all[m*k+i]);
                                vchh = _mm256_loadu_pd(&chh[m*j+i]);
                                vchl = _mm256_loadu_pd(&chl[m*j+i]);
                                vclh = _mm256_loadu_pd(&clh[m*j+i]);
                                vcll = _mm256_loadu_pd(&cll[m*j+i]);
                                
                                //calc
                                _d_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vbhh,vahh,vahl,valh,vall);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                //store
                                _mm256_storeu_pd( &chh[m*j+i], vchh );
                                _mm256_storeu_pd( &chl[m*j+i], vchl );
                                _mm256_storeu_pd( &clh[m*j+i], vclh );
                                _mm256_storeu_pd( &cll[m*j+i], vcll );
                            }
                            sbhh = bhh[l*j+k];
                            for (i = m-p; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                sahl = ahl[m*k+i];
                                salh = alh[m*k+i];
                                sall = all[m*k+i];
                                
                                //calc
                                _d_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sahh,sahl,salh,sall);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
            }
            break;
    }
}
//-----------------------------------------------------------------
// Quad-Double - Double-Double operation
//-----------------------------------------------------------------

/**
 *
 * @fn _qd_scl_dd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the scaler of double-double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh,bhl input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _qd_scl_dd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *alh,double *all,double *bhh,double *bhl, int m, int n, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sahl,sbhh,sbhl,salh,sall,schh,schl,sclh,scll;
                    sahh = ahh[0];
                    sahl = ahl[0];
                    salh = alh[0];
                    sall = all[0];
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        
                        //calc
                        _dd_mul_qd(&schh,&schl,&sclh,&scll,sbhh,sbhl,sahh,sahl,salh,sall);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahh,vahl,vbhh,vbhl,valh,vall,vchh,vchl,vclh,vcll;
                    double vzhh[4]={0,0,0,0};
                    double vzhl[4]={0,0,0,0};
                    double vzlh[4]={0,0,0,0};
                    double vzll[4]={0,0,0,0};
                    vahh = _mm256_set_pd(ahh[0],0,0,0);
                    vahl = _mm256_set_pd(ahl[0],0,0,0);
                    valh = _mm256_set_pd(alh[0],0,0,0);
                    vall = _mm256_set_pd(all[0],0,0,0);
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vbhh = _mm256_set_pd(bhh[i],0,0,0);
                        vbhl = _mm256_set_pd(bhl[i],0,0,0);
                        
                        //calc
                        _dd_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vbhh,vbhl,vahh,vahl,valh,vall);
                        
                        //store
                        _mm256_storeu_pd( &vzhh[0], vchh );
                        _mm256_storeu_pd( &vzhl[0], vchl );
                        _mm256_storeu_pd( &vzlh[0], vclh );
                        _mm256_storeu_pd( &vzll[0], vcll );
                        
                        chh[i] = vzhh[3];
                        chl[i] = vzhl[3];
                        clh[i] = vzlh[3];
                        cll[i] = vzll[3];
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sahl,sbhh,sbhl,salh,sall,schh,schl,sclh,scll;
                    __m256d vahh,vahl,vbhh,vbhl,valh,vall,vchh,vchl,vclh,vcll;
                    sahh = ahh[0];
                    sahl = ahl[0];
                    salh = alh[0];
                    sall = all[0];
                    vahh = _mm256_broadcast_sd(&ahh[0]);
                    vahl = _mm256_broadcast_sd(&ahl[0]);
                    valh = _mm256_broadcast_sd(&alh[0]);
                    vall = _mm256_broadcast_sd(&all[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        
                        //calc
                        _dd_mul_qd_avx2(&vchh,&vchl,&vclh,&vcll,vbhh,vbhl,vahh,vahl,valh,vall);
                                                
                        //store
                        _mm256_storeu_pd( &chh[i], vchh );
                        _mm256_storeu_pd( &chl[i], vchl );
                        _mm256_storeu_pd( &clh[i], vclh );
                        _mm256_storeu_pd( &cll[i], vcll );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        
                        //calc
                        _dd_mul_qd(&schh,&schl,&sclh,&scll,sbhh,sbhl,sahh,sahl,salh,sall);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sahl,sbhh,sbhl,salh,sall,schh,schl,sclh,scll;
                    __m256d vahh,vahl,vbhh,vbhl,valh,vall,vchh,vchl,vclh,vcll;
                    sahh = ahh[0];
                    sahl = ahl[0];
                    salh = alh[0];
                    sall = all[0];
                    vahh = _mm256_broadcast_sd(&ahh[0]);
                    vahl = _mm256_broadcast_sd(&ahl[0]);
                    valh = _mm256_broadcast_sd(&alh[0]);
                    vall = _mm256_broadcast_sd(&all[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        
                        //calc
                        _dd_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vbhh,vbhl,vahh,vahl,valh,vall);
                                                
                        //store
                        _mm256_storeu_pd( &chh[i], vchh );
                        _mm256_storeu_pd( &chl[i], vchl );
                        _mm256_storeu_pd( &clh[i], vclh );
                        _mm256_storeu_pd( &cll[i], vcll );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        
                        //calc
                        _dd_mul_qd(&schh,&schl,&sclh,&scll,sbhh,sbhl,sahh,sahl,salh,sall);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
            }
    }
}
/**
 *
 * @fn _qd_mv_dd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-vector multiplicaion of quad-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh,bhl input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _qd_mv_dd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *alh,double *all,double *bhh,double *bhl, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
                    
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sahl,sbhh,sbhl,salh,sall,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        sbhh = bhh[k];
                        sbhl = bhl[k];
                        for (i = 0; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            //calc
                            _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sbhl,sahh,sahl,salh,sall);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                            
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int k,i;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double vte1[4]={0,0,0,0};
                    double vte2[4]={0,0,0,0};
                    double vte3[4]={0,0,0,0};
                    double vte4[4]={0,0,0,0};
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double schh,schl,sclh,scll;
                    __m256d vahh,vahl,vbhh,vbhl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_set_pd(bhh[k],0,0,0);
                        vbhl = _mm256_set_pd(bhl[k],0,0,0);
                        for (i = 0; i < m; i++) {
                            //load
                            vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                            vahl = _mm256_set_pd(ahl[m*k+i],0,0,0);
                            valh = _mm256_set_pd(alh[m*k+i],0,0,0);
                            vall = _mm256_set_pd(all[m*k+i],0,0,0);
                            vzhh = _mm256_set_pd(tchh[m*tid+i],0,0,0);
                            vzhl = _mm256_set_pd(tchl[m*tid+i],0,0,0);
                            vzlh = _mm256_set_pd(tclh[m*tid+i],0,0,0);
                            vzll = _mm256_set_pd(tcll[m*tid+i],0,0,0);
                            
                            //calc
                            _dd_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vbhh,vbhl,vahh,vahl,valh,vall);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            //store
                            _mm256_storeu_pd( &vte1[0], vchh );
                            _mm256_storeu_pd( &vte2[0], vchl );
                            _mm256_storeu_pd( &vte3[0], vclh );
                            _mm256_storeu_pd( &vte4[0], vcll );
                            
                            tchh[m*tid+i]=vte1[3];
                            tchl[m*tid+i]=vte2[3];
                            tclh[m*tid+i]=vte3[3];
                            tcll[m*tid+i]=vte4[3];
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sahl,sbhh,sbhl,salh,sall,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,vbhh,vbhl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_broadcast_sd(&bhh[k]);
                        vbhl = _mm256_broadcast_sd(&bhl[k]);
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            valh = _mm256_loadu_pd(&alh[m*k+i]);
                            vall = _mm256_loadu_pd(&all[m*k+i]);
                            vzhh = _mm256_loadu_pd(&tchh[m*tid+i]);
                            vzhl = _mm256_loadu_pd(&tchl[m*tid+i]);
                            vzlh = _mm256_loadu_pd(&tclh[m*tid+i]);
                            vzll = _mm256_loadu_pd(&tcll[m*tid+i]);
                            
                            //calc
                            _dd_mul_qd_avx2(&vchh,&vchl,&vclh,&vcll,vbhh,vbhl,vahh,vahl,valh,vall);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            
                            //store
                            _mm256_storeu_pd( &tchh[m*tid+i], vchh );
                            _mm256_storeu_pd( &tchl[m*tid+i], vchl );
                            _mm256_storeu_pd( &tclh[m*tid+i], vclh );
                            _mm256_storeu_pd( &tcll[m*tid+i], vcll );
                        }
                        sbhh = bhh[k];
                        sbhl = bhl[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            
                            //calc
                            _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sbhl,sahh,sahl,salh,sall);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sahl,sbhh,sbhl,salh,sall,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,vbhh,vbhl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_broadcast_sd(&bhh[k]);
                        vbhl = _mm256_broadcast_sd(&bhl[k]);
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            valh = _mm256_loadu_pd(&alh[m*k+i]);
                            vall = _mm256_loadu_pd(&all[m*k+i]);
                            vzhh = _mm256_loadu_pd(&tchh[m*tid+i]);
                            vzhl = _mm256_loadu_pd(&tchl[m*tid+i]);
                            vzlh = _mm256_loadu_pd(&tclh[m*tid+i]);
                            vzll = _mm256_loadu_pd(&tcll[m*tid+i]);
                            
                            //calc
                            _dd_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vbhh,vbhl,vahh,vahl,valh,vall);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            
                            //store
                            _mm256_storeu_pd( &tchh[m*tid+i], vchh );
                            _mm256_storeu_pd( &tchl[m*tid+i], vchl );
                            _mm256_storeu_pd( &tclh[m*tid+i], vclh );
                            _mm256_storeu_pd( &tcll[m*tid+i], vcll );
                        }
                        sbhh = bhh[k];
                        sbhl = bhl[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            
                            //calc
                            _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sbhl,sahh,sahl,salh,sall);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
            }
            break;
    }
}
/**
 *
 * @fn _qd_tmv_dd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the transposed matrix-vector multiplicaion of quad-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh,bhl input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _qd_tmv_dd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *alh,double *all,double *bhh,double *bhl, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    double schh,schl,sclh,scll,sahh,sahl,sbhh,sbhl,salh,sall,szhh,szhl,szlh,szll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        schh=0;
                        schl=0;
                        sclh=0;
                        scll=0;
                        for (i = 0; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            sbhh = bhh[i];
                            sbhl = bhl[i];
                            
                            //calc
                            _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sbhl,sahh,sahl,salh,sall);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,schh,schl,sclh,scll);
                            
                            //store
                            schh = szhh;
                            schl = szhl;
                            sclh = szlh;
                            scll = szll;
                        }
                        chh[k] = schh;
                        chl[k] = schl;
                        clh[k] = sclh;
                        cll[k] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    __m256d vahh,vahl,vbhh,vbhl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m; i++) {
                            //load
                            vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                            vahl = _mm256_set_pd(ahl[m*k+i],0,0,0);
                            valh = _mm256_set_pd(alh[m*k+i],0,0,0);
                            vall = _mm256_set_pd(all[m*k+i],0,0,0);
                            vbhh = _mm256_set_pd(bhh[i],0,0,0);
                            vbhl = _mm256_set_pd(bhl[i],0,0,0);
                            
                            //calc
                            _dd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vbhh,vbhl,vahh,vahl,valh,vall);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        
                        chh[k] = tchh[3];
                        chl[k] = tchl[3];
                        clh[k] = tclh[3];
                        cll[k] = tcll[3];
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    double schh,schl,sclh,scll,sahh,sahl,sbhh,sbhl,salh,sall,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,vbhh,vbhl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            valh = _mm256_loadu_pd(&alh[m*k+i]);
                            vall = _mm256_loadu_pd(&all[m*k+i]);
                            vbhh = _mm256_loadu_pd(&bhh[i]);
                            vbhl = _mm256_loadu_pd(&bhl[i]);
                            
                            //calc
                            _dd_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vbhh,vbhl,vahh,vahl,valh,vall);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            sbhh = bhh[i];
                            sbhl = bhl[i];
                            
                            //calc
                            _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sbhl,sahh,sahl,salh,sall);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        for(i=1;i<4;i++){
                            //calc
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[i],tchl[i],tclh[i],tcll[i],tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        chh[k] = tchh[0];
                        chl[k] = tchl[0];
                        clh[k] = tclh[0];
                        cll[k] = tcll[0];
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    double schh,schl,sclh,scll,sahh,sahl,sbhh,sbhl,salh,sall,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,vbhh,vbhl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            valh = _mm256_loadu_pd(&alh[m*k+i]);
                            vall = _mm256_loadu_pd(&all[m*k+i]);
                            vbhh = _mm256_loadu_pd(&bhh[i]);
                            vbhl = _mm256_loadu_pd(&bhl[i]);
                            
                            //calc
                            _dd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vbhh,vbhl,vahh,vahl,valh,vall);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            sbhh = bhh[i];
                            sbhl = bhl[i];
                            
                            //calc
                            _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sbhl,sahh,sahl,salh,sall);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        for(i=1;i<4;i++){
                            //calc
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[i],tchl[i],tclh[i],tcll[i],tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        chh[k] = tchh[0];
                        chl[k] = tchl[0];
                        clh[k] = tclh[0];
                        cll[k] = tcll[0];
                    }
                }
                break;
            }
            break;
    }
}
/**
 *
 * @fn _qd_mm_dd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-matrix multiplicaion of quad-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh,bhl input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _qd_mm_dd(double* chh,double* chl,double* clh,double* cll,double *bhh,double *bhl, double *ahh,double *ahl,double *alh,double *all,int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    double sahh,sahl,sbhh,sbhl,salh,sall,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            sbhh = bhh[l*j+k];
                            sbhl = bhl[l*j+k];
                            for (i = 0; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                sahl = ahl[m*k+i];
                                salh = alh[m*k+i];
                                sall = all[m*k+i];
                                
                                //calc
                                _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sbhl,sahh,sahl,salh,sall);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    __m256d vahh,vahl,vbhh,vbhl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
                    double szhh[4]={0,0,0,0};
                    double szhl[4]={0,0,0,0};
                    double szlh[4]={0,0,0,0};
                    double szll[4]={0,0,0,0};
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_set_pd(bhh[l*j+k],0,0,0);
                            vbhl = _mm256_set_pd(bhl[l*j+k],0,0,0);
                            
                            for (i = 0; i < m; i++) {
                                vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                                vahl = _mm256_set_pd(ahl[m*k+i],0,0,0);
                                valh = _mm256_set_pd(alh[m*k+i],0,0,0);
                                vall = _mm256_set_pd(all[m*k+i],0,0,0);
                                vchh = _mm256_set_pd(chh[m*j+i],0,0,0);
                                vchl = _mm256_set_pd(chl[m*j+i],0,0,0);
                                vclh = _mm256_set_pd(clh[m*j+i],0,0,0);
                                vcll = _mm256_set_pd(cll[m*j+i],0,0,0);
                                
                                //calc
                                _dd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vbhh,vbhl,vahh,vahl,valh,vall);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                _mm256_storeu_pd( &szhh[0], vchh );
                                _mm256_storeu_pd( &szhl[0], vchl );
                                _mm256_storeu_pd( &szlh[0], vclh );
                                _mm256_storeu_pd( &szll[0], vcll );
                                
                                chh[m*j+i] = szhh[3];
                                chl[m*j+i] = szhl[3];
                                clh[m*j+i] = szlh[3];
                                cll[m*j+i] = szll[3];
                            }
                        }
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    int p=m%4;
                    double sahh,sahl,sbhh,sbhl,salh,sall,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,vbhh,vbhl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_broadcast_sd(&bhh[l*j+k]);
                            vbhl = _mm256_broadcast_sd(&bhl[l*j+k]);
                            
                            for (i = 0; i < m-p; i+=4) {
                                //load
                                vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                                vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                                valh = _mm256_loadu_pd(&alh[m*k+i]);
                                vall = _mm256_loadu_pd(&all[m*k+i]);
                                vchh = _mm256_loadu_pd(&chh[m*j+i]);
                                vchl = _mm256_loadu_pd(&chl[m*j+i]);
                                vclh = _mm256_loadu_pd(&clh[m*j+i]);
                                vcll = _mm256_loadu_pd(&cll[m*j+i]);
                                
                                //calc
                                _dd_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vbhh,vbhl,vahh,vahl,valh,vall);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                //store
                                _mm256_storeu_pd( &chh[m*j+i], vchh );
                                _mm256_storeu_pd( &chl[m*j+i], vchl );
                                _mm256_storeu_pd( &clh[m*j+i], vclh );
                                _mm256_storeu_pd( &cll[m*j+i], vcll );
                            }
                            sbhh = bhh[l*j+k];
                            sbhl = bhl[l*j+k];
                            for (i = m-p; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                sahl = ahl[m*k+i];
                                salh = alh[m*k+i];
                                sall = all[m*k+i];
                                
                                //calc
                                _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sbhl,sahh,sahl,salh,sall);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    int p=m%4;
                    double sahh,sahl,sbhh,sbhl,salh,sall,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,vbhh,vbhl,valh,vall,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_broadcast_sd(&bhh[l*j+k]);
                            vbhl = _mm256_broadcast_sd(&bhl[l*j+k]);
                            
                            for (i = 0; i < m-p; i+=4) {
                                //load
                                vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                                vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                                valh = _mm256_loadu_pd(&alh[m*k+i]);
                                vall = _mm256_loadu_pd(&all[m*k+i]);
                                vchh = _mm256_loadu_pd(&chh[m*j+i]);
                                vchl = _mm256_loadu_pd(&chl[m*j+i]);
                                vclh = _mm256_loadu_pd(&clh[m*j+i]);
                                vcll = _mm256_loadu_pd(&cll[m*j+i]);
                                
                                //calc
                                _dd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vbhh,vbhl,vahh,vahl,valh,vall);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                //store
                                _mm256_storeu_pd( &chh[m*j+i], vchh );
                                _mm256_storeu_pd( &chl[m*j+i], vchl );
                                _mm256_storeu_pd( &clh[m*j+i], vclh );
                                _mm256_storeu_pd( &cll[m*j+i], vcll );
                            }
                            sbhh = bhh[l*j+k];
                            sbhl = bhl[l*j+k];
                            for (i = m-p; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                sahl = ahl[m*k+i];
                                salh = alh[m*k+i];
                                sall = all[m*k+i];
                                
                                //calc
                                _dd_mul_qd(&szhh,&szhl,&szlh,&szll,sbhh,sbhl,sahh,sahl,salh,sall);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
            }
            break;
    }
}
//-----------------------------------------------------------------
// Quad-Double - Quad-Double operation
//-----------------------------------------------------------------

/**
 *
 * @fn _qd_a_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int)
 * @brief You can compute the addition of quad-double and quad-double numbers in a vector or matrix. You can choose the number of OpenMP threads and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _qd_a_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *alh,double *all,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int omp_threadNum, int avx)
{
    int i;
    int p = m % 4;
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    switch(avx){
        case 0:
#pragma omp parallel num_threads(omp_threadNum)
        {
            double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
#pragma omp for
            for (i = 0; i < m * n; i ++){
                //load
                sahh = ahh[i];
                sahl = ahl[i];
                salh = alh[i];
                sall = all[i];
                sbhh = bhh[i];
                sbhl = bhl[i];
                sblh = blh[i];
                sbll = bll[i];
                
                //calc
                _qd_add_qd(&schh,&schl,&sclh,&scll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                
                //store
                chh[i] = schh;
                chl[i] = schl;
                clh[i] = sclh;
                cll[i] = scll;
            }
        }
        break;
        case 1:
#pragma omp parallel num_threads(omp_threadNum)
        {
            double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
            __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll;
#pragma omp for
            for (i = 0; i < m * n - p; i += 4){
                //load
                vahh = _mm256_loadu_pd(&ahh[i]);
                vahl = _mm256_loadu_pd(&ahl[i]);
                valh = _mm256_loadu_pd(&alh[i]);
                vall = _mm256_loadu_pd(&all[i]);
                vbhh = _mm256_loadu_pd(&bhh[i]);
                vbhl = _mm256_loadu_pd(&bhl[i]);
                vblh = _mm256_loadu_pd(&blh[i]);
                vbll = _mm256_loadu_pd(&bll[i]);
                
                //calc
                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                
                //store
                _mm256_storeu_pd( &chh[i], vchh );
                _mm256_storeu_pd( &chl[i], vchl );
                _mm256_storeu_pd( &clh[i], vclh );
                _mm256_storeu_pd( &cll[i], vcll );
            }
#pragma omp for
            for (i = m * n - p; i < m * n; i ++){
                //load
                sahh = ahh[i];
                sahl = ahl[i];
                salh = alh[i];
                sall = all[i];
                sbhh = bhh[i];
                sbhl = bhl[i];
                sblh = blh[i];
                sbll = bll[i];
                
                //calc
                _qd_add_qd(&schh,&schl,&sclh,&scll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                
                //store
                chh[i] = schh;
                chl[i] = schl;
                clh[i] = sclh;
                cll[i] = scll;
            }
        }
        break;
    }
}


/**
 *
 * @fn _qd_scl_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the scaler of quad-double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _qd_scl_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *alh,double *all,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
                    sahh = ahh[0];
                    sahl = ahl[0];
                    salh = alh[0];
                    sall = all[0];
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _qd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll;
                    double vzhh[4]={0,0,0,0};
                    double vzhl[4]={0,0,0,0};
                    double vzlh[4]={0,0,0,0};
                    double vzll[4]={0,0,0,0};
                    vahh = _mm256_set_pd(ahh[0],0,0,0);
                    vahl = _mm256_set_pd(ahl[0],0,0,0);
                    valh = _mm256_set_pd(alh[0],0,0,0);
                    vall = _mm256_set_pd(all[0],0,0,0);
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vbhh = _mm256_set_pd(bhh[i],0,0,0);
                        vbhl = _mm256_set_pd(bhl[i],0,0,0);
                        vblh = _mm256_set_pd(blh[i],0,0,0);
                        vbll = _mm256_set_pd(bll[i],0,0,0);
                        
                        //calc
                        _qd_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                        
                        //store
                        _mm256_storeu_pd( &vzhh[0], vchh );
                        _mm256_storeu_pd( &vzhl[0], vchl );
                        _mm256_storeu_pd( &vzlh[0], vclh );
                        _mm256_storeu_pd( &vzll[0], vcll );                                      
                        
                        chh[i] = vzhh[3];
                        chl[i] = vzhl[3];
                        clh[i] = vzlh[3];
                        cll[i] = vzll[3];
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll;
                    sahh = ahh[0];
                    sahl = ahl[0];
                    salh = alh[0];
                    sall = all[0];
                    vahh = _mm256_broadcast_sd(&ahh[0]);
                    vahl = _mm256_broadcast_sd(&ahl[0]);
                    valh = _mm256_broadcast_sd(&alh[0]);
                    vall = _mm256_broadcast_sd(&all[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _qd_mul_qd_avx2(&vchh,&vchl,&vclh,&vcll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                                                
                        //store
                        _mm256_storeu_pd( &chh[i], vchh );
                        _mm256_storeu_pd( &chl[i], vchl );
                        _mm256_storeu_pd( &clh[i], vclh );
                        _mm256_storeu_pd( &cll[i], vcll );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _qd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll;
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll;
                    sahh = ahh[0];
                    sahl = ahl[0];
                    salh = alh[0];
                    sall = all[0];
                    vahh = _mm256_broadcast_sd(&ahh[0]);
                    vahl = _mm256_broadcast_sd(&ahl[0]);
                    valh = _mm256_broadcast_sd(&alh[0]);
                    vall = _mm256_broadcast_sd(&all[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _qd_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                                                
                        //store
                        _mm256_storeu_pd( &chh[i], vchh );
                        _mm256_storeu_pd( &chl[i], vchl );
                        _mm256_storeu_pd( &clh[i], vclh );
                        _mm256_storeu_pd( &cll[i], vcll );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _qd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
            }
    }
}

/**
 *
 * @fn _qd_axpy_qd(double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the axpy of double-double and double-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chi,clo output (double-double precision)
 * @param ahi,alo input (double-double precision)
 * @param bhi,blo input (double-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 * @param fma  ON/OFF of FMA
 *
 */
void _qd_axpy_qd(double *chh, double *chl,double *clh, double *cll,double* xhh, double* xhl, double* xlh, double* xll,  double *ahh,double *ahl,  double *alh,double *all,double *bhh, double *bhl,double *blh, double *bll,int m, int n, int omp_threadNum, int avx, int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
// #pragma omp parallel num_threads(omp_threadNum)
//                 {
//                     double sahi,salo,sbhi,sblo,schi,sclo;
//                     sahi = ahi[0];
//                     salo = alo[0];
// #pragma omp for
//                     for (i = 0; i < m * n; i ++){
//                         //load
//                         sbhi = bhi[i];
//                         sblo = blo[i];
//                         
//                         //calc
//                         _dd_mul_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
//                         
//                         //store
//                         chi[i] = schi;
//                         clo[i] = sclo;
//                     }
//                 }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vxhh,vxhl,vxlh,vxll;
                    double vzhh[4]={0,0,0,0};
                    double vzhl[4]={0,0,0,0};
                    double vzlh[4]={0,0,0,0};
                    double vzll[4]={0,0,0,0};
                    vxhh = _mm256_set_pd(xhh[0],0,0,0);
                    vxhl = _mm256_set_pd(xhl[0],0,0,0);
                    vxlh = _mm256_set_pd(xlh[0],0,0,0);
                    vxll = _mm256_set_pd(xll[0],0,0,0);
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vahh = _mm256_set_pd(ahh[i],0,0,0);
                        vahl = _mm256_set_pd(ahl[i],0,0,0);
                        valh = _mm256_set_pd(alh[i],0,0,0);
                        vall = _mm256_set_pd(all[i],0,0,0);
                        vbhh = _mm256_set_pd(bhh[i],0,0,0);
                        vbhl = _mm256_set_pd(bhl[i],0,0,0);
                        vblh = _mm256_set_pd(blh[i],0,0,0);
                        vbll = _mm256_set_pd(bll[i],0,0,0);
                        
                        //calc
                        _qd_mul_qd_fma(&vahh,&vahl,&valh,&vall,vxhh,vxhl,vxlh,vxll,vahh,vahl,valh,vall);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                        
                                               //store
                        _mm256_storeu_pd( &chh[i], vchh );
                        _mm256_storeu_pd( &chl[i], vchl );
                        _mm256_storeu_pd( &clh[i], vclh );
                        _mm256_storeu_pd( &cll[i], vcll );
                        
                        chh[i]=vzhh[3];
                        chl[i]=vzhl[3];
                        clh[i]=vzlh[3];
                        cll[i]=vzll[3];
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
// #pragma omp parallel num_threads(omp_threadNum)
//                 {
//                     double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,sxhh,sxhl,sxlh,sxll;
//                     __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vxhh,vxhl,vxlh,vxll;
//                     sxhi = xhi[0];
//                     sxlo = xlo[0];
//                     vxhi = _mm256_broadcast_sd(&xhi[0]);
//                     vxlo = _mm256_broadcast_sd(&xlo[0]);
// #pragma omp for schedule(static)
//                     for (i = 0; i < m * n - p; i += 4) {
//                         //load
//                         vahi = _mm256_loadu_pd(&ahi[i]);
//                         valo = _mm256_loadu_pd(&alo[i]);
//                         vbhi = _mm256_loadu_pd(&bhi[i]);
//                         vblo = _mm256_loadu_pd(&blo[i]);
//                         
//                         //calc
//                         _dd_mul_dd_avx2(&vahi,&valo,vxhi,vxlo,vahi,valo);
//                         _dd_add_dd_avx2(&vchi,&vclo,vahi,valo,vbhi,vblo);
//                         
//                         //store
//                         _mm256_storeu_pd( &chi[i], vchi );
//                         _mm256_storeu_pd( &clo[i], vclo );
//                         
//                     }
// #pragma omp for
//                     for (i = m * n - p; i < m * n; i ++){
//                         //load
//                         sahi = ahi[i];
//                         salo = alo[i];
//                         sbhi = bhi[i];
//                         sblo = blo[i];
//                         
//                         //calc
//                         _dd_mul_dd(&sahi,&salo,sxhi,sxlo,sahi,salo);
//                         _dd_add_dd(&schi,&sclo,sahi,salo,sbhi,sblo);
//                         
//                         //store
//                         chi[i] = schi;
//                         clo[i] = sclo;
//                     }
//                 }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,sxhh,sxhl,sxlh,sxll;
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vxhh,vxhl,vxlh,vxll;
                    sxhh = xhh[0];
                    sxhl = xhl[0];
                    sxlh = xlh[0];
                    sxll = xll[0];
                    vxhh = _mm256_broadcast_sd(&xhh[0]);
                    vxhl = _mm256_broadcast_sd(&xhl[0]);
                    vxlh = _mm256_broadcast_sd(&xlh[0]);
                    vxll = _mm256_broadcast_sd(&xll[0]);
#pragma omp for schedule(static)
                    for (i = 0; i < m * n - p; i += 4) {
                        //load
                        vahh = _mm256_loadu_pd(&ahh[i]);
                        vahl = _mm256_loadu_pd(&ahl[i]);
                        valh = _mm256_loadu_pd(&alh[i]);
                        vall = _mm256_loadu_pd(&all[i]);
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _qd_mul_qd_fma(&vahh,&vahl,&valh,&vall,vxhh,vxhl,vxlh,vxll,vahh,vahl,valh,vall);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                        
                        
                        //store
                        _mm256_storeu_pd( &chh[i], vchh );
                        _mm256_storeu_pd( &chl[i], vchl );
                        _mm256_storeu_pd( &clh[i], vclh );
                        _mm256_storeu_pd( &cll[i], vcll );
                        
                    }
#pragma omp for
                    for (i = m * n - p; i < m * n; i ++){
                        //load
                        sahh = ahh[i];
                        sahl = ahl[i];
                        salh = alh[i];
                        sall = all[i];
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _qd_mul_qd(&sahh,&sahl,&salh,&sall,sxhh,sxhl,sxlh,sxll,sahh,sahl,salh,sall);
                        _qd_add_qd(&schh,&schl,&sclh,&scll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                        
                        //store
                        chh[i] = schh;
                        chl[i] = schl;
                        clh[i] = sclh;
                        cll[i] = scll;
                    }
                }
                break;
            }
    }
}


/**
 *
 * @fn _qd_dot_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the inner product of quad-double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _qd_dot_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *alh,double *all,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads];
                    double tchl [num_threads];
                    double tclh [num_threads];
                    double tcll [num_threads];
                    for (i = 0; i < num_threads; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sahh = ahh[i];
                        sahl = ahl[i];
                        salh = alh[i];
                        sall = all[i];
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _qd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                        _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[tid],tchl[tid],tclh[tid],tcll[tid],schh,schl,sclh,scll);
                        
                        //store
                        tchh[tid] = szhh;
                        tchl[tid] = szhl;
                        tclh[tid] = szlh;
                        tcll[tid] = szll;
                    }
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[tid],tchl[tid],tclh[tid],tcll[tid],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads*4];
                    double tchl [num_threads*4];
                    double tclh [num_threads*4];
                    double tcll [num_threads*4];
                    for (i = 0; i < num_threads*4; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vzhh,vzhl,vzlh,vzll;
                    __m256d vchh = _mm256_setzero_pd();
                    __m256d vchl = _mm256_setzero_pd();
                    __m256d vclh = _mm256_setzero_pd();
                    __m256d vcll = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vahh = _mm256_set_pd(ahh[i],0,0,0);
                        vahl = _mm256_set_pd(ahl[i],0,0,0);
                        valh = _mm256_set_pd(alh[i],0,0,0);
                        vall = _mm256_set_pd(all[i],0,0,0);
                        vbhh = _mm256_set_pd(bhh[i],0,0,0);
                        vbhl = _mm256_set_pd(bhl[i],0,0,0);
                        vblh = _mm256_set_pd(blh[i],0,0,0);
                        vbll = _mm256_set_pd(bll[i],0,0,0);
                        
                        //calc
                        _qd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        
                    }
                    //store
                    _mm256_storeu_pd( &tchh[4*tid], vchh );
                    _mm256_storeu_pd( &tchl[4*tid], vchl );
                    _mm256_storeu_pd( &tclh[4*tid], vclh );
                    _mm256_storeu_pd( &tcll[4*tid], vcll );
                    
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid+3],tchl[4*tid+3],tclh[4*tid+3],tcll[4*tid+3],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads*4];
                    double tchl [num_threads*4];
                    double tclh [num_threads*4];
                    double tcll [num_threads*4];
                    for (i = 0; i < num_threads*4; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vzhh,vzhl,vzlh,vzll;
                    __m256d vchh = _mm256_setzero_pd();
                    __m256d vchl = _mm256_setzero_pd();
                    __m256d vclh = _mm256_setzero_pd();
                    __m256d vcll = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahh = _mm256_loadu_pd(&ahh[i]);
                        vahl = _mm256_loadu_pd(&ahl[i]);
                        valh = _mm256_loadu_pd(&alh[i]);
                        vall = _mm256_loadu_pd(&all[i]);
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _qd_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        
                    }
                    //store
                    _mm256_storeu_pd( &tchh[4*tid], vchh );
                    _mm256_storeu_pd( &tchl[4*tid], vchl );
                    _mm256_storeu_pd( &tclh[4*tid], vclh );
                    _mm256_storeu_pd( &tcll[4*tid], vcll );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        
                        //load
                        sahh = ahh[i];
                        sahl = ahl[i];
                        salh = alh[i];
                        sall = all[i];
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _qd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                        _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],schh,schl,sclh,scll);
                        
                        //store
                        tchh[4*tid] = szhh;
                        tchl[4*tid] = szhl;
                        tclh[4*tid] = szlh;
                        tcll[4*tid] = szll;
                        
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid+i],tchl[4*tid+i],tclh[4*tid+i],tcll[4*tid+i],tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid]);
                        
                        //store
                        tchh[4*tid] = schh;
                        tchl[4*tid] = schl;
                        tclh[4*tid] = sclh;
                        tcll[4*tid] = scll;
                    }
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads*4];
                    double tchl [num_threads*4];
                    double tclh [num_threads*4];
                    double tcll [num_threads*4];
                    for (i = 0; i < num_threads*4; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vzhh,vzhl,vzlh,vzll;
                    __m256d vchh = _mm256_setzero_pd();
                    __m256d vchl = _mm256_setzero_pd();
                    __m256d vclh = _mm256_setzero_pd();
                    __m256d vcll = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahh = _mm256_loadu_pd(&ahh[i]);
                        vahl = _mm256_loadu_pd(&ahl[i]);
                        valh = _mm256_loadu_pd(&alh[i]);
                        vall = _mm256_loadu_pd(&all[i]);
                        vbhh = _mm256_loadu_pd(&bhh[i]);
                        vbhl = _mm256_loadu_pd(&bhl[i]);
                        vblh = _mm256_loadu_pd(&blh[i]);
                        vbll = _mm256_loadu_pd(&bll[i]);
                        
                        //calc
                        _qd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        
                    }
                    //store
                    _mm256_storeu_pd( &tchh[4*tid], vchh );
                    _mm256_storeu_pd( &tchl[4*tid], vchl );
                    _mm256_storeu_pd( &tclh[4*tid], vclh );
                    _mm256_storeu_pd( &tcll[4*tid], vcll );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        
                        //load
                        sahh = ahh[i];
                        sahl = ahl[i];
                        salh = alh[i];
                        sall = all[i];
                        sbhh = bhh[i];
                        sbhl = bhl[i];
                        sblh = blh[i];
                        sbll = bll[i];
                        
                        //calc
                        _qd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                        _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],schh,schl,sclh,scll);
                        
                        //store
                        tchh[4*tid] = szhh;
                        tchl[4*tid] = szhl;
                        tclh[4*tid] = szlh;
                        tcll[4*tid] = szll;
                        
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid+i],tchl[4*tid+i],tclh[4*tid+i],tcll[4*tid+i],tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid]);
                        
                        //store
                        tchh[4*tid] = schh;
                        tchl[4*tid] = schl;
                        tclh[4*tid] = sclh;
                        tcll[4*tid] = scll;
                    }
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                    }
                }
                break;
            }
    }
}

/**
 *
 * @fn _qd_dot_qd_self(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int)
 * @brief You can compute the inner product of quad-double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _qd_dot_qd_self(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *alh,double *all, int m, int n, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0)  omp_threadNum = omp_get_max_threads();
    int i;
    int p = m % 4;
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads];
                    double tchl [num_threads];
                    double tclh [num_threads];
                    double tcll [num_threads];
                    for (i = 0; i < num_threads; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        sahh = ahh[i];
                        sahl = ahl[i];
                        salh = alh[i];
                        sall = all[i];
                        
                        //calc
                        _qd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,salh,sall,sahh,sahl,salh,sall);
                        _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[tid],tchl[tid],tclh[tid],tcll[tid],schh,schl,sclh,scll);
                        
                        //store
                        tchh[tid] = szhh;
                        tchl[tid] = szhl;
                        tclh[tid] = szlh;
                        tcll[tid] = szll;
                    }
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[tid],tchl[tid],tclh[tid],tcll[tid],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads*4];
                    double tchl [num_threads*4];
                    double tclh [num_threads*4];
                    double tcll [num_threads*4];
                    for (i = 0; i < num_threads*4; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vzhh,vzhl,vzlh,vzll;
                    __m256d vchh = _mm256_setzero_pd();
                    __m256d vchl = _mm256_setzero_pd();
                    __m256d vclh = _mm256_setzero_pd();
                    __m256d vcll = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n; i ++){
                        //load
                        vahh = _mm256_set_pd(ahh[i],0,0,0);
                        vahl = _mm256_set_pd(ahl[i],0,0,0);
                        valh = _mm256_set_pd(alh[i],0,0,0);
                        vall = _mm256_set_pd(all[i],0,0,0);
                        
                        //calc
                        _qd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,valh,vall,vahh,vahl,valh,vall);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        
                    }
                    //store
                    _mm256_storeu_pd( &tchh[4*tid], vchh );
                    _mm256_storeu_pd( &tchl[4*tid], vchl );
                    _mm256_storeu_pd( &tclh[4*tid], vclh );
                    _mm256_storeu_pd( &tcll[4*tid], vcll );
                    
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid+3],tchl[4*tid+3],tclh[4*tid+3],tcll[4*tid+3],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                        
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads*4];
                    double tchl [num_threads*4];
                    double tclh [num_threads*4];
                    double tcll [num_threads*4];
                    for (i = 0; i < num_threads*4; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vzhh,vzhl,vzlh,vzll;
                    __m256d vchh = _mm256_setzero_pd();
                    __m256d vchl = _mm256_setzero_pd();
                    __m256d vclh = _mm256_setzero_pd();
                    __m256d vcll = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahh = _mm256_loadu_pd(&ahh[i]);
                        vahl = _mm256_loadu_pd(&ahl[i]);
                        valh = _mm256_loadu_pd(&alh[i]);
                        vall = _mm256_loadu_pd(&all[i]);
                        
                        //calc
                        _qd_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,valh,vall,vahh,vahl,valh,vall);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        
                    }
                    //store
                    _mm256_storeu_pd( &tchh[4*tid], vchh );
                    _mm256_storeu_pd( &tchl[4*tid], vchl );
                    _mm256_storeu_pd( &tclh[4*tid], vclh );
                    _mm256_storeu_pd( &tcll[4*tid], vcll );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        
                        //load
                        sahh = ahh[i];
                        sahl = ahl[i];
                        salh = alh[i];
                        sall = all[i];
                        
                        //calc
                        _qd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,salh,sall,sahh,sahl,salh,sall);
                        _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],schh,schl,sclh,scll);
                        
                        //store
                        tchh[4*tid] = szhh;
                        tchl[4*tid] = szhl;
                        tclh[4*tid] = szlh;
                        tcll[4*tid] = szll;
                        
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid+i],tchl[4*tid+i],tclh[4*tid+i],tcll[4*tid+i],tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid]);
                        
                        //store
                        tchh[4*tid] = schh;
                        tchl[4*tid] = schl;
                        tclh[4*tid] = sclh;
                        tcll[4*tid] = scll;
                    }
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll,schh,schl,sclh,scll;
                    double tchh [num_threads*4];
                    double tchl [num_threads*4];
                    double tclh [num_threads*4];
                    double tcll [num_threads*4];
                    for (i = 0; i < num_threads*4; i++) {
                        tchh[i]=0;
                        tchl[i]=0;
                        tclh[i]=0;
                        tcll[i]=0;
                    }
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vzhh,vzhl,vzlh,vzll;
                    __m256d vchh = _mm256_setzero_pd();
                    __m256d vchl = _mm256_setzero_pd();
                    __m256d vclh = _mm256_setzero_pd();
                    __m256d vcll = _mm256_setzero_pd();
#pragma omp for
                    for (i = 0; i < m * n - p; i += 4){
                        //load
                        vahh = _mm256_loadu_pd(&ahh[i]);
                        vahl = _mm256_loadu_pd(&ahl[i]);
                        valh = _mm256_loadu_pd(&alh[i]);
                        vall = _mm256_loadu_pd(&all[i]);
                        
                        //calc
                        _qd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,valh,vall,vahh,vahl,valh,vall);
                        _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        
                    }
                    //store
                    _mm256_storeu_pd( &tchh[4*tid], vchh );
                    _mm256_storeu_pd( &tchl[4*tid], vchl );
                    _mm256_storeu_pd( &tclh[4*tid], vclh );
                    _mm256_storeu_pd( &tcll[4*tid], vcll );
#pragma omp for
                    for (i = m - p; i < m * n; i++) {
                        
                        //load
                        sahh = ahh[i];
                        sahl = ahl[i];
                        salh = alh[i];
                        sall = all[i];
                        
                        //calc
                        _qd_mul_qd(&schh,&schl,&sclh,&scll,sahh,sahl,salh,sall,sahh,sahl,salh,sall);
                        _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],schh,schl,sclh,scll);
                        
                        //store
                        tchh[4*tid] = szhh;
                        tchl[4*tid] = szhl;
                        tclh[4*tid] = szlh;
                        tcll[4*tid] = szll;
                        
                    }
#pragma omp critical
                    for(i=1;i<4;i++){
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid+i],tchl[4*tid+i],tclh[4*tid+i],tcll[4*tid+i],tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid]);
                        
                        //store
                        tchh[4*tid] = schh;
                        tchl[4*tid] = schl;
                        tclh[4*tid] = sclh;
                        tcll[4*tid] = scll;
                    }
#pragma omp critical
                    {
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,tchh[4*tid],tchl[4*tid],tclh[4*tid],tcll[4*tid],chh[0],chl[0],clh[0],cll[0]);
                        
                        //store
                        chh[0] = schh;
                        chl[0] = schl;
                        clh[0] = sclh;
                        cll[0] = scll;
                    }
                }
                break;
            }
    }
}

/**
 *
 * @fn _qd_mv_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-vector multiplicaion of quad-double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _qd_mv_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *alh,double *all,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
                    
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        sbhh = bhh[k];
                        sbhl = bhl[k];
                        sblh = blh[k];
                        sbll = bll[k];
                        for (i = 0; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            //calc
                            _qd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                            
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int k,i;
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double vte1[4]={0,0,0,0};
                    double vte2[4]={0,0,0,0};
                    double vte3[4]={0,0,0,0};
                    double vte4[4]={0,0,0,0};
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double schh,schl,sclh,scll;
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_set_pd(bhh[k],0,0,0);
                        vbhl = _mm256_set_pd(bhl[k],0,0,0);
                        vblh = _mm256_set_pd(blh[k],0,0,0);
                        vbll = _mm256_set_pd(bll[k],0,0,0);
                        for (i = 0; i < m; i++) {
                            //load
                            vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                            vahl = _mm256_set_pd(ahl[m*k+i],0,0,0);
                            valh = _mm256_set_pd(alh[m*k+i],0,0,0);
                            vall = _mm256_set_pd(all[m*k+i],0,0,0);
                            vzhh = _mm256_set_pd(tchh[m*tid+i],0,0,0);
                            vzhl = _mm256_set_pd(tchl[m*tid+i],0,0,0);
                            vzlh = _mm256_set_pd(tclh[m*tid+i],0,0,0);
                            vzll = _mm256_set_pd(tcll[m*tid+i],0,0,0);
                            
                            //calc
                            _qd_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            //store
                            _mm256_storeu_pd( &vte1[0], vchh );
                            _mm256_storeu_pd( &vte2[0], vchl );
                            _mm256_storeu_pd( &vte3[0], vclh );
                            _mm256_storeu_pd( &vte4[0], vcll );
                            
                            tchh[m*tid+i]=vte1[3];
                            tchl[m*tid+i]=vte2[3];
                            tclh[m*tid+i]=vte3[3];
                            tcll[m*tid+i]=vte4[3];
                        }
                    }
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_broadcast_sd(&bhh[k]);
                        vbhl = _mm256_broadcast_sd(&bhl[k]);
                        vblh = _mm256_broadcast_sd(&blh[k]);
                        vbll = _mm256_broadcast_sd(&bll[k]);
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            valh = _mm256_loadu_pd(&alh[m*k+i]);
                            vall = _mm256_loadu_pd(&all[m*k+i]);
                            vzhh = _mm256_loadu_pd(&tchh[m*tid+i]);
                            vzhl = _mm256_loadu_pd(&tchl[m*tid+i]);
                            vzlh = _mm256_loadu_pd(&tclh[m*tid+i]);
                            vzll = _mm256_loadu_pd(&tcll[m*tid+i]);
                            
                            //calc
                            _qd_mul_qd_avx2(&vchh,&vchl,&vclh,&vcll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            //store
                            _mm256_storeu_pd( &tchh[m*tid+i], vchh );
                            _mm256_storeu_pd( &tchl[m*tid+i], vchl );
                            _mm256_storeu_pd( &tclh[m*tid+i], vclh );
                            _mm256_storeu_pd( &tcll[m*tid+i], vcll );
                        }
                        sbhh = bhh[k];
                        sbhl = bhl[k];
                        sblh = blh[k];
                        sbll = bll[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            
                            //calc
                            _qd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    
                    int num_threads=omp_get_num_threads();
                    int tid = omp_get_thread_num();
                    double tchh [num_threads*m];
                    double tchl [num_threads*m];
                    double tclh [num_threads*m];
                    double tcll [num_threads*m];
                    for (k = 0; k < num_threads*m; k++) {
                        tchh[k]=0;
                        tchl[k]=0;
                        tclh[k]=0;
                        tcll[k]=0;
                    }
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for schedule(static)
                    for (k = 0; k < l; k++) {
                        vbhh = _mm256_broadcast_sd(&bhh[k]);
                        vbhl = _mm256_broadcast_sd(&bhl[k]);
                        vblh = _mm256_broadcast_sd(&blh[k]);
                        vbll = _mm256_broadcast_sd(&bll[k]);
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            valh = _mm256_loadu_pd(&alh[m*k+i]);
                            vall = _mm256_loadu_pd(&all[m*k+i]);
                            vzhh = _mm256_loadu_pd(&tchh[m*tid+i]);
                            vzhl = _mm256_loadu_pd(&tchl[m*tid+i]);
                            vzlh = _mm256_loadu_pd(&tclh[m*tid+i]);
                            vzll = _mm256_loadu_pd(&tcll[m*tid+i]);
                            
                            //calc
                            _qd_mul_qd_fma(&vchh,&vchl,&vclh,&vcll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                            
                            
                            //store
                            _mm256_storeu_pd( &tchh[m*tid+i], vchh );
                            _mm256_storeu_pd( &tchl[m*tid+i], vchl );
                            _mm256_storeu_pd( &tclh[m*tid+i], vclh );
                            _mm256_storeu_pd( &tcll[m*tid+i], vcll );
                        }
                        sbhh = bhh[k];
                        sbhl = bhl[k];
                        sblh = blh[k];
                        sbll = bll[k];
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            
                            //calc
                            _qd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&schh,&schl,&sclh,&scll,szhh,szhl,szlh,szll,tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                            
                            //store
                            tchh[m*tid+i]=schh;
                            tchl[m*tid+i]=schl;
                            tclh[m*tid+i]=sclh;
                            tcll[m*tid+i]=scll;
                        }
                    }
                    
#pragma omp critical
                    for(i=0;i<m;i++){
                        
                        //calc
                        _qd_add_qd(&schh,&schl,&sclh,&scll,chh[i],chl[i],clh[i],cll[i],tchh[m*tid+i],tchl[m*tid+i],tclh[m*tid+i],tcll[m*tid+i]);
                        
                        //store
                        chh[i]=schh;
                        chl[i]=schl;
                        clh[i]=sclh;
                        cll[i]=scll;
                    }
                    
                }
                
                break;
            }
            break;
    }
}
/**
 *
 * @fn _qd_tmv_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the transposed matrix-vector multiplicaion of quad-double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _qd_tmv_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *alh,double *all,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    double schh,schl,sclh,scll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        schh=0;
                        schl=0;
                        sclh=0;
                        scll=0;
                        for (i = 0; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            sbhh = bhh[i];
                            sbhl = bhl[i];
                            sblh = blh[i];
                            sbll = bll[i];
                            
                            //calc
                            _qd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,schh,schl,sclh,scll);
                                                        
                            //store
                            schh = szhh;
                            schl = szhl;
                            sclh = szlh;
                            scll = szll;
                        }
                        chh[k] = schh;
                        chl[k] = schl;
                        clh[k] = sclh;
                        cll[k] = scll;
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m; i++) {
                            //load
                            vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                            vahl = _mm256_set_pd(ahl[m*k+i],0,0,0);
                            valh = _mm256_set_pd(alh[m*k+i],0,0,0);
                            vall = _mm256_set_pd(all[m*k+i],0,0,0);
                            vbhh = _mm256_set_pd(bhh[i],0,0,0);
                            vbhl = _mm256_set_pd(bhl[i],0,0,0);
                            vblh = _mm256_set_pd(blh[i],0,0,0);
                            vbll = _mm256_set_pd(bll[i],0,0,0);
                            
                            //calc
                            _qd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        
                        chh[k] = tchh[3];
                        chl[k] = tchl[3];
                        clh[k] = tclh[3];
                        cll[k] = tcll[3];
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    double schh,schl,sclh,scll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            valh = _mm256_loadu_pd(&alh[m*k+i]);
                            vall = _mm256_loadu_pd(&all[m*k+i]);
                            vbhh = _mm256_loadu_pd(&bhh[i]);
                            vbhl = _mm256_loadu_pd(&bhl[i]);
                            vblh = _mm256_loadu_pd(&blh[i]);
                            vbll = _mm256_loadu_pd(&bll[i]);
                            
                            //calc
                            _qd_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            sbhh = bhh[i];
                            sbhl = bhl[i];
                            sblh = blh[i];
                            sbll = bll[i];
                            
                            //calc
                            _qd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        for(i=1;i<4;i++){
                            //calc
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[i],tchl[i],tclh[i],tcll[i],tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        chh[k] = tchh[0];
                        chl[k] = tchl[0];
                        clh[k] = tclh[0];
                        cll[k] = tcll[0];
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int p = m%4;
                    int k,i;
                    double tchh[4]={0,0,0,0};
                    double tchl[4]={0,0,0,0};
                    double tclh[4]={0,0,0,0};
                    double tcll[4]={0,0,0,0};
                    double schh,schl,sclh,scll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (k = 0; k < l; k++) {
                        vchh = _mm256_setzero_pd();
                        vchl = _mm256_setzero_pd();
                        vclh = _mm256_setzero_pd();
                        vcll = _mm256_setzero_pd();
                        for (i = 0; i < m-p; i+=4) {
                            //load
                            vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                            vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                            valh = _mm256_loadu_pd(&alh[m*k+i]);
                            vall = _mm256_loadu_pd(&all[m*k+i]);
                            vbhh = _mm256_loadu_pd(&bhh[i]);
                            vbhl = _mm256_loadu_pd(&bhl[i]);
                            vblh = _mm256_loadu_pd(&blh[i]);
                            vbll = _mm256_loadu_pd(&bll[i]);
                            
                            //calc
                            _qd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                            _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                        }
                        _mm256_storeu_pd( &tchh[0], vchh );
                        _mm256_storeu_pd( &tchl[0], vchl );
                        _mm256_storeu_pd( &tclh[0], vclh );
                        _mm256_storeu_pd( &tcll[0], vcll );
                        for (i = m-p; i < m; i++) {
                            //load
                            sahh = ahh[m*k+i];
                            sahl = ahl[m*k+i];
                            salh = alh[m*k+i];
                            sall = all[m*k+i];
                            sbhh = bhh[i];
                            sbhl = bhl[i];
                            sblh = blh[i];
                            sbll = bll[i];
                            
                            //calc
                            _qd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        for(i=1;i<4;i++){
                            //calc
                            _qd_add_qd(&szhh,&szhl,&szlh,&szll,tchh[i],tchl[i],tclh[i],tcll[i],tchh[0],tchl[0],tclh[0],tcll[0]);
                            
                            //store
                            tchh[0] = szhh;
                            tchl[0] = szhl;
                            tclh[0] = szlh;
                            tcll[0] = szll;
                        }
                        chh[k] = tchh[0];
                        chl[k] = tchl[0];
                        clh[k] = tclh[0];
                        cll[k] = tcll[0];
                    }
                }
                break;
            }
            break;
    }
}
/**
 *
 * @fn _qd_mm_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int)
 * @brief You can compute the matrix-matrix multiplicaion of double-double and quad-double numbers. You can choose the number of OpenMP threads and FMA, and AVX2 on / off.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param ahh,ahl,alh,all input (quad-double precision)
 * @param bhh,bhl,blh,bll input (quad-double precision)
 * @param m,n size
 * @param omp_threadNum the number of threads
 * @param avx  ON/OFF of AVX2
 *
 */
void _qd_mm_qd(double* chh,double* chl,double* clh,double* cll,double *ahh,double *ahl,double *alh,double *all,double *bhh,double *bhl,double *blh,double *bll, int m, int n, int l, int omp_threadNum, int avx,int fma)
{
    if(omp_threadNum == 0){
        omp_threadNum = omp_get_max_threads();
    }
    switch(avx){
        case 0:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int i,k,j;
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            sbhh = bhh[l*j+k];
                            sbhl = bhl[l*j+k];
                            sblh = blh[l*j+k];
                            sbll = bll[l*j+k];
                            for (i = 0; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                sahl = ahl[m*k+i];
                                salh = alh[m*k+i];
                                sall = all[m*k+i];
                                
                                //calc
                                _qd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
                    double szhh[4]={0,0,0,0};
                    double szhl[4]={0,0,0,0};
                    double szlh[4]={0,0,0,0};
                    double szll[4]={0,0,0,0};
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_set_pd(bhh[l*j+k],0,0,0);
                            vbhl = _mm256_set_pd(bhl[l*j+k],0,0,0);
                            vblh = _mm256_set_pd(blh[l*j+k],0,0,0);
                            vbll = _mm256_set_pd(bll[l*j+k],0,0,0);
                            
                            for (i = 0; i < m; i++) {
                                vahh = _mm256_set_pd(ahh[m*k+i],0,0,0);
                                vahl = _mm256_set_pd(ahl[m*k+i],0,0,0);
                                valh = _mm256_set_pd(alh[m*k+i],0,0,0);
                                vall = _mm256_set_pd(all[m*k+i],0,0,0);
                                vchh = _mm256_set_pd(chh[m*j+i],0,0,0);
                                vchl = _mm256_set_pd(chl[m*j+i],0,0,0);
                                vclh = _mm256_set_pd(clh[m*j+i],0,0,0);
                                vcll = _mm256_set_pd(cll[m*j+i],0,0,0);
                                
                                //calc
                                _qd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                _mm256_storeu_pd( &szhh[0], vchh );
                                _mm256_storeu_pd( &szhl[0], vchl );
                                _mm256_storeu_pd( &szlh[0], vclh );
                                _mm256_storeu_pd( &szll[0], vcll );
                                
                                chh[m*j+i] = szhh[3];
                                chl[m*j+i] = szhl[3];
                                clh[m*j+i] = szlh[3];
                                cll[m*j+i] = szll[3];
                            }
                        }
                    }
                }
                break;
            }
            break;
        case 1:
            switch(fma){
                case 0:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    int p=m%4;
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_broadcast_sd(&bhh[l*j+k]);
                            vbhl = _mm256_broadcast_sd(&bhl[l*j+k]);
                            vblh = _mm256_broadcast_sd(&blh[l*j+k]);
                            vbll = _mm256_broadcast_sd(&bll[l*j+k]);
                            
                            for (i = 0; i < m-p; i+=4) {
                                //load
                                vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                                vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                                valh = _mm256_loadu_pd(&alh[m*k+i]);
                                vall = _mm256_loadu_pd(&all[m*k+i]);
                                vchh = _mm256_loadu_pd(&chh[m*j+i]);
                                vchl = _mm256_loadu_pd(&chl[m*j+i]);
                                vclh = _mm256_loadu_pd(&clh[m*j+i]);
                                vcll = _mm256_loadu_pd(&cll[m*j+i]);
                                
                                //calc
                                _qd_mul_qd_avx2(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                //store
                                _mm256_storeu_pd( &chh[m*j+i], vchh );
                                _mm256_storeu_pd( &chl[m*j+i], vchl );
                                _mm256_storeu_pd( &clh[m*j+i], vclh );
                                _mm256_storeu_pd( &cll[m*j+i], vcll );
                            }
                            sbhh = bhh[l*j+k];
                            sbhl = bhl[l*j+k];
                            sblh = blh[l*j+k];
                            sbll = bll[l*j+k];
                            for (i = m-p; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                sahl = ahl[m*k+i];
                                salh = alh[m*k+i];
                                sall = all[m*k+i];
                                
                                //calc
                                _qd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
                case 1:
#pragma omp parallel num_threads(omp_threadNum)
                {
                    int j,k,i;
                    int p=m%4;
                    double sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll,schh,schl,sclh,scll,szhh,szhl,szlh,szll;
                    __m256d vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll,vchh,vchl,vclh,vcll,vzhh,vzhl,vzlh,vzll;
#pragma omp for
                    for (j = 0; j < n; j++) {
                        for (k = 0; k < l; k++) {
                            vbhh = _mm256_broadcast_sd(&bhh[l*j+k]);
                            vbhl = _mm256_broadcast_sd(&bhl[l*j+k]);
                            vblh = _mm256_broadcast_sd(&blh[l*j+k]);
                            vbll = _mm256_broadcast_sd(&bll[l*j+k]);
                            
                            for (i = 0; i < m-p; i+=4) {
                                //load
                                vahh = _mm256_loadu_pd(&ahh[m*k+i]);
                                vahl = _mm256_loadu_pd(&ahl[m*k+i]);
                                valh = _mm256_loadu_pd(&alh[m*k+i]);
                                vall = _mm256_loadu_pd(&all[m*k+i]);
                                vchh = _mm256_loadu_pd(&chh[m*j+i]);
                                vchl = _mm256_loadu_pd(&chl[m*j+i]);
                                vclh = _mm256_loadu_pd(&clh[m*j+i]);
                                vcll = _mm256_loadu_pd(&cll[m*j+i]);
                                
                                //calc
                                _qd_mul_qd_fma(&vzhh,&vzhl,&vzlh,&vzll,vahh,vahl,valh,vall,vbhh,vbhl,vblh,vbll);
                                _qd_add_qd_avx2(&vchh,&vchl,&vclh,&vcll,vzhh,vzhl,vzlh,vzll,vchh,vchl,vclh,vcll);
                                
                                //store
                                _mm256_storeu_pd( &chh[m*j+i], vchh );
                                _mm256_storeu_pd( &chl[m*j+i], vchl );
                                _mm256_storeu_pd( &clh[m*j+i], vclh );
                                _mm256_storeu_pd( &cll[m*j+i], vcll );
                            }
                            sbhh = bhh[l*j+k];
                            sbhl = bhl[l*j+k];
                            sblh = blh[l*j+k];
                            sbll = bll[l*j+k];
                            for (i = m-p; i < m; i++) {
                                //load
                                sahh = ahh[m*k+i];
                                sahl = ahl[m*k+i];
                                salh = alh[m*k+i];
                                sall = all[m*k+i];
                                
                                //calc
                                _qd_mul_qd(&szhh,&szhl,&szlh,&szll,sahh,sahl,salh,sall,sbhh,sbhl,sblh,sbll);
                                _qd_add_qd(&szhh,&szhl,&szlh,&szll,szhh,szhl,szlh,szll,chh[m*j+i],chl[m*j+i],clh[m*j+i],cll[m*j+i]);
                                
                                //store
                                chh[m*j+i] = szhh;
                                chl[m*j+i] = szhl;
                                clh[m*j+i] = szlh;
                                cll[m*j+i] = szll;
                            }
                        }
                    }
                }
                break;
            }
            break;
    }
}
//-----------------------------------------------------------------
// Definition of base operation
//-----------------------------------------------------------------

//-----------------------------------------------------------------
// Serial
//-----------------------------------------------------------------

/**
 *
 * @fn _d_add_dd(double*,double*,double,double,double)
 * @brief The function compute the addition of double and double-double number with serial computing.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (double-double precision)
 *
 */
void _d_add_dd(double* chi,double* clo,double ahi,double bhi,double blo)
{
    double s,v,eh,el,chi_n,clo_n;
    //threeSum2(ahi, bhi, blo);
    //twosum(ahi,bhi)
    s = ahi + bhi;
    v = s - ahi;
    eh = (ahi - (s - v)) + (bhi - v);
    //twosum(s,blo)
    chi_n = s + blo;
    v = chi_n - s;
    el = (s - (chi_n - v)) + (blo - v);
    
    clo_n = eh + el;
    
    *chi = chi_n;
    *clo = clo_n;
}
/**
 *
 * @fn _d_mul_dd(double*,double*,double,double,double)
 * @brief The function compute the multiplication of double and double-double number with serial computing.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (double-double precision)
 *
 */
void _d_mul_dd(double* chi,double* clo,double ahi,double bhi,double blo)
{
    double t,t1,ah,al,bh,bl,p1,p2,chi_n,clo_n;
    //twoprod(ahi,bhi)
    //split(ahi)
    t = 134217729 * ahi;
    ah = t - (t - ahi);
    al = ahi - ah;
    //split(bhi)
    t1 = 134217729 * bhi;
    bh = t1 - (t1 - bhi);
    bl = bhi - bh;
    
    p1 = ahi * bhi;
    p2 = ((ah * bh - p1) + ah * bl + al * bh) + al * bl;
    p2 = p2 + ahi * blo;
    
    chi_n = p1 + p2;
    clo_n = p2 - (chi_n - p1);
    
    *chi = chi_n;
    *clo = clo_n;
}
/**
 *
 * @fn _d_add_qd(double*,double*,double*,double*,double,double,double,double,double)
 * @brief The function compute the addition of double and quad-double number with serial computing.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param xhh input (double precision)
 * @param yhh,yhl,ylh,yll input (quad-double precision)
 *
 */
void _d_add_qd(double* chh,double* chl,double* clh,double* cll,double xhh,double yhh,double yhl,double ylh,double yll)
{
    double s1,s2,s3,s4,s5,e,v,s,ss,t1,t2,t3,t4,zhh,zhl,zlh,zll;
    //twoSum(x,y.hh);
    s1 = xhh + yhh;
    v = s1 - xhh;
    e = (xhh - (s1 - v)) + (yhh - v);
    
    //twoSum(e,y.hl);
    s2 = e + yhl;
    v = s2 - e;
    e = (e - (s2 - v)) + (yhl - v);
    
    //twoSum(e,y.lh);;
    s3 = e + ylh;
    v = s3 - e;
    e = (e - (s3 - v)) + (ylh - v);
    
    //two_sum(e, y.ll)
    s4 = e + yll;
    v = s4 - e;
    s5 = (e - (s4 - v)) + (yll - v);
    
    //renormalize(s1, s2, s3, s4, s5)
    s = s4 + s5;
    t4 = s5 - (s - s4);
    ss = s3 + s;
    t3 = s - (ss - s3);
    s  = s2 + ss;
    t2 = ss - (s - s2);
    zhh = s1 + s;
    t1 = s - (zhh - s1);
    s = t3 + t4;
    t3 = t4 - (s - t3);
    ss = t2 + s;
    t2 = s - (ss - t2);
    zhl = t1 + ss;
    t1 = ss - (zhl - t1);
    s = t2 + t3;
    t2 = t3 - (s - t2);
    zlh = t1 + s;
    t1 = s - (zlh - t1);
    zll = t1 + t2;
    
    *chh = zhh;
    *chl = zhl;
    *clh = zlh;
    *cll = zll;
}
/**
 *
 * @fn _d_mul_qd(double*,double*,double*,double*,double,double,double,double,double)
 * @brief The function compute the multiplication of double and quad-double number with serial computing.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param xhh input (double precision)
 * @param yhh,yhl,ylh,yll input (quad-double precision)
 *
 */
void _d_mul_qd(double* chh,double* chl,double* clh,double* cll,double xhh,double yhh,double yhl,double ylh,double yll)
{
    
    double p1,p2,p3,p4,p5,s,ss,ah,al,bh,bl,e1,e2,e3,e4,v,sh,eh,t1,t2,t3,t4,zhh,zhl,zlh,zll;
    //mul
    //two_prod(xhh, yhh);
    p1 = xhh * yhh;
    //[ah, al] = split(a);
    ah = 134217729 * xhh;
    ah = ah - (ah - xhh);
    al = xhh - ah;
    //[bh, bl] = split(b);
    bh = 134217729 * yhh;
    bh = bh - (bh - yhh);
    bl = yhh - bh;
    
    e1 = ((ah * bh - p1) + ah * bl + al * bh) + al * bl;
    
    
    //two_prod(xhh, yhh);
    p2 = xhh * yhl;
    
    //[bh, bl] = split(b);
    bh = 134217729 * yhl;
    bh = bh - (bh - yhl);
    bl = yhl - bh;
    
    e2 = ((ah * bh - p2) + ah * bl + al * bh) + al * bl;
    
    //two_prod(xhh, yhh);
    p3 = xhh * ylh;
    
    //[bh, bl] = split(b);
    bh = 134217729 * ylh;
    bh = bh - (bh - ylh);
    bl = ylh - bh;
    
    e3 = ((ah * bh - p3) + ah * bl + al * bh) + al * bl;
    
    p4 = xhh * yll;
    
    //twosum(p2, e1)
    s = p2 + e1;
    v = s - p2;
    e1 = (p2 - (s - v)) + (e1 - v);
    
    p2 = s;
    
    //[p3,e2,e1]=three_sum(p3, e2, e1);
    //two_Sum(p3, e2);
    s = p3 + e2;
    v = s - p3;
    eh = (p3 - (s - v)) + (e2 - v);
    //two_Sum(s, e1);
    p3 = s + e1;
    v = p3 - s;
    s = (s - (p3 - v)) + (e1 - v);
    //two_Sum(eh, s);
    e2 = eh + s;
    v = e2 - eh;
    e1 = (eh - (e2 - v)) + (s - v);
    
    
    //[p4,e2]=threesum2(p4,e3,e2)
    //twosum(p4,e3)
    s = p4 + e3;
    v = s - p4;
    eh = (p4 - (s - v)) + (e3 - v);
    //twosum(s,e2)
    p4 = s + e2;
    v = p4 - s;
    s = (s - (p4 - v)) + (e2 - v);
    e2 = eh + s;
    
    p5 = e1 + e2;
    
    
    //renormalize(from p1 to p5);
    s = p4 + p5;
    t4 = p5 - (s - p4);
    ss = p3 + s;
    t3 = s - (ss - p3);
    s  = p2 + ss;
    t2 = ss - (s - p2);
    zhh = p1 + s;
    t1 = s - (zhh - p1);
    s = t3 + t4;
    t3 = t4 - (s - t3);
    ss = t2 + s;
    t2 = s - (ss - t2);
    zhl = t1 + ss;
    t1 = ss - (zhl - t1);
    s = t2 + t3;
    t2 = t3 - (s -t2);
    zlh = t1 + s;
    t1 = s - (zlh - t1);
    zll = t1 + t2;
    
    *chh = zhh;
    *chl = zhl;
    *clh = zlh;
    *cll = zll;
}
/**
 *
 * @fn _dd_add_dd(double*,double*,double,double,double,double)
 * @brief The function compute the addition of double-double and double-double number with serial computing.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (quad-double precision)
 *
 */
void _dd_add_dd(double* chi,double* clo,double ahi,double alo,double bhi,double blo)
{
    double sl,s,v,eh,el,chi_n,clo_n;
    s = ahi + bhi;
    v = s - ahi;
    eh = (ahi - (s - v)) + (bhi - v);
    sl = alo + blo;
    eh = eh + sl;
    chi_n = s + eh;
    clo_n = eh - (chi_n - s);
    
    *chi = chi_n;
    *clo = clo_n;
}
/**
 *
 * @fn _dd_add_dd_ieee(double*,double*,double,double,double,double)
 * @brief The function compute the addition of double-double and double-double number with serial computing.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (quad-double precision)
 *
 */
void _dd_add_dd_ieee(double* chi,double* clo,double ahi,double alo,double bhi,double blo)
{
    /* Compute double-double = double-double + double-double */
    double s1,s2,bv,t1,t2,chi_n,clo_n;
    /* Add two high-order words. */
    s1 = ahi + bhi;
    bv = s1 - ahi;
    s2 = ((bhi - bv) + (ahi - (s1 - bv)));
    /* Add two low-order words. */
    t1 = alo + blo;
    bv = t1 - alo;
    t2 = ((blo - bv) + (alo - (t1 - bv)));
    s2 += t1;
    /* Renormalize (s1, s2) to (t1, s2) */
    t1 = s1 + s2;
    s2 = s2 - (t1 - s1);
    t2 += s2;
    /* Renormalize (t1, t2) */
    chi_n = t1 + t2;
    clo_n = t2 - (chi_n - t1);
    
    *chi = chi_n;
    *clo = clo_n;
}

/**
 *
 * @fn _dd_mul_dd(double*,double*,double,double,double,double)
 * @brief The function compute the multiplication of double-double and double-double number with serial computing.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (quad-double precision)
 *
 */
void _dd_mul_dd(double* chi,double* clo,double ahi,double alo,double bhi,double blo)
{
    double t,t1,ah,al,bh,bl,p1,p2,chi_n,clo_n;
    
    t = 134217729 * ahi;
    ah = t - (t - ahi);
    al = ahi - ah;
    t1 = 134217729 * bhi;
    bh = t1 - (t1 - bhi);
    bl = bhi - bh;
    p1 = ahi * bhi;
    p2 = ((ah * bh - p1) + ah * bl + al * bh) + al * bl;
    p2 = p2 + ahi * blo;
    p2 = p2 + alo * bhi;
    chi_n = p1 + p2;
    clo_n = p2 - (chi_n - p1);
    
    *chi = chi_n;
    *clo = clo_n;
}
/**
 *
 * @fn _dd_add_qd(double*,double*,double*,double*,double,double,double,double,double,double)
 * @brief The function compute the addition of double-double and quad-double number with serial computing.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param xhh,xhl input (double-double precision)
 * @param yhh,yhl,ylh,yll input (quad-double precision)
 *
 */
void _dd_add_qd(double* chh,double* chl,double* clh,double* cll,double xhh,double xhl,double yhh,double yhl,double ylh,double yll)
{
    double s1,s2,s3,s4,s5,e1,e2,e3,e4,e5,v,u,s,ss,t1,t2,t3,t4,zhh,zhl,zlh,zll;
    //addition
    //twoSum(x.hh, y.hh);
    s1 = xhh + yhh;
    v = s1 - xhh;
    e1 = (xhh - (s1 - v)) + (yhh - v);
    
    //twoSum(x.hl, y.hl);
    t1 = xhl + yhl;
    v = t1 - xhl;
    e2 = (xhl - (t1 - v)) + (yhl - v);
    
    //twoSum(t1, e1);
    s2 = t1 + e1;
    v = s2 - t1;
    u = (t1 - (s2 - v)) + (e1 - v);
    
    //twoSum(e2, y.lh);
    t2 = e2 + ylh;
    v = t2 - e2;
    e3 = (e2 - (t2 - v)) + (ylh - v);
    
    //twoSum(u, t2);
    s3 = u + t2;
    v = s3 - u;
    u = (u - (s3 - v)) + (t2 - v);
    
    //twoSum(u, e3);
    t3 = u + e3;
    v = t3 - u;
    e4 = (u - (t3 - v)) + (e3 - v);
    
    //twoSum(t3, y.ll);
    s4 = t3 + yll;
    v = s4 - t3;
    e5 = (t3 - (s4 - v)) + (yll - v);
    
    s5 = e4 + e5;
    
    //renormalize(s1, s2, s3, s4, s5)
    s = s4 + s5;
    t4 = s5 - (s - s4);
    ss = s3 + s;
    t3 = s - (ss - s3);
    s  = s2 + ss;
    t2 = ss - (s - s2);
    zhh = s1 + s;
    t1 = s - (zhh - s1);
    s = t3 + t4;
    t3 = t4 - (s - t3);
    ss = t2 + s;
    t2 = s - (ss - t2);
    zhl = t1 + ss;
    t1 = ss - (zhl - t1);
    s = t2 + t3;
    t2 = t3 - (s - t2);
    zlh = t1 + s;
    t1 = s - (zlh - t1);
    zll = t1 + t2;
    
    *chh=zhh;
    *chl=zhl;
    *clh=zlh;
    *cll=zll;
}
/**
 *
 * @fn _dd_mul_qd(double*,double*,double*,double*,double,double,double,double,double,double)
 * @brief The function compute the multiplication of double-double and quad-double number with serial computing.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param xhh,xhl input (double-double precision)
 * @param yhh,yhl,ylh,yll input (quad-double precision)
 *
 */
void _dd_mul_qd(double* chh,double* chl,double* clh,double* cll,double xhh,double xhl,double yhh,double yhl,double ylh,double yll)
{
    double a1h,a1l,a2h,a2l,e5,p1,p2,p3,p4,p5,s,ss,ah,al,bh,bl,e1,e2,e3,e4,v,sh,eh,t1,t2,t3,t4,zhh,zhl,zlh,zll;
    
    //two_prod(xhh, yhh);
    p1 = xhh * yhh;
    //[ah, al] = split(a);
    a1h = 134217729 * xhh;
    a1h = a1h - (a1h - xhh);
    a1l = xhh - a1h;
    //[bh, bl] = split(b);
    bh = 134217729 * yhh;
    bh = bh - (bh - yhh);
    bl = yhh - bh;
    e1 = ((a1h * bh - p1) + a1h * bl + a1l * bh) + a1l * bl;
    
    //two_prod(xhl, yhh);
    p2 = xhl * yhh;
    //[ah, al] = split(a);
    a2h = 134217729 * xhl;
    a2h = a2h - (a2h - xhl);
    a2l = xhl - a2h;
    e2 = ((a2h * bh - p2) + a2h * bl + a2l * bh) + a2l * bl;
    
    //two_prod(xhh, yhl);
    p3 = xhh * yhl;
    //[bh, bl] = split(b);
    bh = 134217729 * yhl;
    bh = bh - (bh - yhl);
    bl = yhl - bh;
    e3 = ((a1h * bh - p3) + a1h * bl + a1l * bh) + a1l * bl;
    
    //two_prod(xhl, yhl);
    p4 = xhl * yhl;
    e4 = ((a2h * bh - p4) + a2h * bl + a2l * bh) + a2l * bl;
    
    //two_prod(xhh, ylh);
    p5 = xhh * ylh;
    //[bh, bl] = split(b);
    bh = 134217729 * ylh;
    bh = bh - (bh - ylh);
    bl = ylh - bh;
    e5 = ((a1h * bh - p5) + a1h * bl + a1l * bh) + a1l * bl;
    
    //[p3,e2,e1]=three_sum(p2, p3, e1);
    //two_Sum(p3, p3);
    s = p2 + p3;
    v = s - p2;
    eh = (p2 - (s - v)) + (p3 - v);
    //two_Sum(s, e1);
    p2 = s + e1;
    v = p2 - s;
    s = (s - (p2 - v)) + (e1 - v);
    //two_Sum(eh, s);
    p3 = eh + s;
    v = p3 - eh;
    e1 = (eh - (p3 - v)) + (s - v);
    
    //three_sum(p3, p4, p5);
    //two_Sum(p3, p4);
    s = p3 + p4;
    v = s - p3;
    eh = (p3 - (s - v)) + (p4 - v);
    //two_Sum(s, p5);
    p3 = s + p5;
    v = p3 - s;
    s = (s - (p3 - v)) + (p5 - v);
    //two_Sum(eh, s);
    p4 = eh + s;
    v = p4 - eh;
    p5 = (eh - (p4 - v)) + (s - v);
    
    
    //twosum(e2, e3)
    s = e2 + e3;
    v = s - e2;
    e3 = (e2 - (s - v)) + (e3 - v);
    e2 = s;
    
    //twosum(p3, e2)
    s = p3 + e2;
    v = s - p3;
    e2 = (p3 - (s - v)) + (e2 - v);
    p3 = s;
    
    //twosum(p4, e3)
    s = p4 + e3;
    v = s - p4;
    e3 = (p4 - (s - v)) + (e3 - v);
    p4 = s;
    
    //twosum(p4, e2)
    s = p4 + e2;
    v = s - p4;
    e2 = (p4 - (s - v)) + (e2 - v);
    p4 = s;
    
    p5 = p5 + e2 + e3;
    
    ss = xhl * ylh + xhh * yll + e4 + e5;
    
    //threesum2(p4,e1,ss)
    //twosum(p4,e1)
    s = p4 + e1;
    v = s - p4;
    eh = (p4 - (s - v)) + (e1 - v);
    //twosum(s,ss)
    p4 = s + ss;
    v = p4 - s;
    s = (s - (p4 - v)) + (ss - v);
    e1 = eh + s;
    
    p5 = p5 + e1;
    
    //renormalize(from p1 to p5);
    s = p4 + p5;
    t4 = p5 - (s - p4);
    ss = p3 + s;
    t3 = s - (ss - p3);
    s  = p2 + ss;
    t2 = ss - (s - p2);
    zhh = p1 + s;
    t1 = s - (zhh - p1);
    s = t3 + t4;
    t3 = t4 - (s - t3);
    ss = t2 + s;
    t2 = s - (ss - t2);
    zhl = t1 + ss;
    t1 = ss - (zhl - t1);
    s = t2 + t3;
    t2 = t3 - (s -t2);
    zlh = t1 + s;
    t1 = s - (zlh - t1);
    zll = t1 + t2;
    
    *chh=zhh;
    *chl=zhl;
    *clh=zlh;
    *cll=zll;
}
/**
 *
 * @fn _qd_add_qd(double*,double*,double*,double*,double,double,double,double,double,double,double,double)
 * @brief The function compute the addition of quad-double and quad-double number with serial computing.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param xhh,xhl,xlh,xll input (quad-double precision)
 * @param yhh,yhl,ylh,yll input (quad-double precision)
 *
 */
void _qd_add_qd(double* chh,double* chl,double* clh,double* cll,double xhh,double xhl,double xlh,double xll,double yhh,double yhl,double ylh,double yll)
{
    double e1,e2,e3,e4,v,sh,eh,t1,t2,t3,t4,zhh,zhl,zlh,zll;
    
    //two_sum(&c0, &e1, a0, b0); 12
    zhh = xhh + yhh;
    v = zhh - xhh;
    e1 = (xhh - (zhh - v)) + (yhh - v);
    
    //two_sum(&c1, &e2, a1, b1);
    zhl = xhl + yhl;
    v = zhl - xhl;
    e2 = (xhl - (zhl - v)) + (yhl - v);
    
    //two_sum(&c1, &e1, c1, e1);
    e3 = zhl + e1;
    v = e3 - zhl;
    e1 = (zhl - (e3 - v)) + (e1 - v);
    zhl=e3;
    
    //two_sum(&c2, &e3, a2, b2); 48
    zlh = xlh + ylh;
    v = zlh - xlh;
    e3 = (xlh - (zlh - v)) + (ylh - v);
    //three_sum(&c2, &e1, &e2, c2, e2, e1);
    sh = zlh + e2;
    v = sh - zlh;
    eh = (zlh - (sh - v)) + (e2 - v);
    
    zlh = sh + e1;
    v = zlh - sh;
    sh = (sh - (zlh - v)) + (e1 - v);
    
    e1 = eh + sh;
    v = e1 - eh;
    e2 = (eh - (e1 - v)) + (sh - v);
    
    //two_sum(&c3, &e4, a3, b3); 96
    zll = xll + yll;
    v = zll - xll;
    e4 = (xll - (zll - v)) + (yll - v);
    
    //three_sum2(&c3, &e1, c3, e3, e1);125
    sh = zll + e3;
    v = sh - zll;
    eh = (zll - (sh - v)) + (e3 - v);
    zll = sh + e1;
    v = zll - sh;
    
    e1 = eh + (sh - (zll - v)) + (e1 - v);
    e1 = e1 + e2 + e4;
    eh=zhh;
    e2=zhl;
    e3=zlh;
    e4=zll;
//        renormalize(&c0, &c1, &c2, &c3, c0, c1, c2, c3, e1); 129+69 = 198
    sh = e4 + e1;
    t4 = e1 - (sh - e4);
    v = e3 + sh;
    t3 = sh - (v - e3);
    sh  = e2 + v;
    t2 = v - (sh - e2);
    zhh = eh + sh;
    t1 = sh - (zhh - eh);
    sh = t3 + t4;
    t3 = t4 - (sh - t3);
    v = t2 + sh;
    t2 = sh - (v - t2);
    zhl = t1 + v;
    t1 = v - (zhl - t1);
    sh = t2 + t3;
    t2 = t3 - (sh -t2);
    zlh = t1 + sh;
    t1 = sh - (zlh - t1);
    zll = t1 + t2;
    
    *chh = zhh;
    *chl = zhl;
    *clh = zlh;
    *cll = zll;
}
/**
 *
 * @fn _qd_mul_qd(double*,double*,double*,double*,double,double,double,double,double,double,double,double)
 * @brief The function compute the multiplication of quad-double and quad-double number with serial computing.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param xhh,xhl,xlh,xll input (quad-double precision)
 * @param yhh,yhl,ylh,yll input (quad-double precision)
 *
 */
void _qd_mul_qd(double* chh,double* chl,double* clh,double* cll,double xhh,double xhl,double xlh,double xll,double yhh,double yhl,double ylh,double yll)
{
    double bhh,bhl,blh,bll;
    double p01,p10,p11,p02,p20,ah00,al00,bh00,bl00,ah01,al01,bh01,bl01,ah10,al10;
    double bh10,bl10,ah11,al11,bh11,bl11,ah20,al20,bh20,bl20,ah02,al02,bh02,bl02;
//two_prod(zhh, al00, xhh, yhh);
    bhh = xhh * yhh;
    //[ah, al] = Calc.split(a);3
    ah00 = 134217729 * xhh;
    ah00 = ah00 - (ah00 - xhh);
    al00 = xhh - ah00;
    //[bh, bl] = Calc.split(b);9*2+3+10=31
    bh00 = 134217729 * yhh;
    bh00 = bh00 - (bh00 - yhh);
    bl00 = yhh - bh00;
    al00 = ((ah00 * bh00 - bhh) + ah00 * bl00 + al00 * bh00) + al00* bl00;
    
    
    //O(eps) terms
    //two_prod(p01, bl00, xhh, yhl);
    p01=xhh*yhl;
    //split
    ah01=134217729 * xhh;
    ah01=ah01-(ah01-xhh);
    al01=xhh-ah01;
    //split
    bh01=134217729 * yhl;
    bh01=bh01-(bh01-yhl);
    bl01=yhl-bh01;
    bl00 = ((ah01 * bh01 - p01) + ah01 * bl01 + al01 * bh01) + al01 * bl01;
    
    //two_prod(&p10, &al10, xhl, yhh);
    p10=xhl*yhh;
    //split
    ah10=134217729 * xhl;
    ah10=ah10-(ah10-xhl);
    al10=xhl-ah10;
    //split
    bh10=134217729 * yhh;
    bh10=bh10-(bh10-yhh);
    bl10=yhh-bh10;
    
    al10 = ((ah10 * bh10 - p10) + ah10 * bl10 + al10 * bh10) + al10 * bl10;
    
    //three_sum(zhl, p10, p01, p01, p10, al00);
    // [ah00, bh00] = Calc.twoSum(p01, p10);12
    ah00=p01+p10;
    al01=ah00-p01;
    bh00=(p01-(ah00-al01))+(p10-al01);
    // [zhl, ah00] = Calc.twoSum(sh, al00);24
    bhl=ah00+al00;
    al01=bhl-ah00;
    ah00=(ah00-(bhl-al01))+(al00-al01);
    // [p10, p01] = Calc.twoSum(eh, sh);36
    p10=bh00+ah00;
    al01=p10-bh00;
    p01=(bh00-(p10-al01))+(ah00-al01);
    
    
    //---------------------------------------------------
    //O(eps^2) terms
    //two_prod(&p02, &ah02, xhh, ylh);
    p02=xhh*ylh;
    //split
    ah02=134217729 * xhh;
    ah02=ah02-(ah02-xhh);
    al02=xhh-ah02;
    //split
    bh02=134217729 * ylh;
    bh02=bh02-(bh02-ylh);
    bl02=ylh-bh02;
    
    ah02 = ((ah02 * bh02 - p02) + ah02 * bl02 + al02 * bh02) + al02 * bl02;
    //two_prod(&p11, &bl11, xhl, yhl);
    p11=xhl*yhl;
    //split
    ah11=134217729 * xhl;
    ah11=ah11-(ah11-xhl);
    al11=xhl-ah11;
    //split
    bh11=134217729 * yhl;
    bh11=bh11-(bh11-yhl);
    bl11=yhl-bh11;
    
    bl11 = ((ah11 * bh11 - p11) + ah11 * bl11 + al11 * bh11) + al11 * bl11;
    //two_prod(&p20, &al11, xlh, yhh);
    p20=xlh*yhh;
    //split
    ah20=134217729 * xlh;
    ah20=ah20-(ah20-xlh);
    al20=xlh-ah20;
    //split
    bh20=134217729 * yhh;
    bh20=bh20-(bh20-yhh);
    bl20=yhh-bh20;
    
    al11 = (ah20 * bh20 - p20) + ah20 * bl20 + al20 * bh20 + al20 * bl20;
    
    
    //six three sum for p10, bl00, al10, p02, p11, p20
    
    //three_sum(&p10, &bl00, &al10, p10, bl00, al10);
    // [ah20, bh20] = Calc.twoSum(p10, bl00);
    ah20=p10+bl00;
    al20=ah20-p10;
    bh20=(p10-(ah20-al20))+(bl00-al20);
    // [p10, ah20] = Calc.twoSum(ah20, al10);
    p10=ah20+al10;
    al20=p10-ah20;
    ah20=(ah20-(p10-al20))+(al10-al20);
    // [bl00, al10] = Calc.twoSum(bh20, ah20);
    bl00=bh20+ah20;
    al20=bl00-bh20;
    al10=(bh20-(bl00-al20))+(ah20-al20);
    
    
    //three_sum(&p02, &p11, &p20, p02, p11, p20);
    // [ah11, bh02] = Calc.twoSum(p02, p11);
    ah11=p02+p11;
    al02=ah11-p02;
    bh02=(p02-(ah11-al02))+(p11-al02);
    // [p02, ah11] = Calc.twoSum(ah11, p20);
    p02=ah11+p20;
    al02=p02-ah11;
    ah11=(ah11-(p02-al02))+(p20-al02);
    // [p11, p20] = Calc.twoSum(bh02, ah11);
    p11=bh02+bh02;
    al02=p11-bh02;
    p20=(bh02-(p11-al02))+(ah11-al02);
    
    // two_sum(zlh, &p10, p02, p10);
    blh = p02 + p10;
    bl02 = blh - p02;
    p10 = (p02 - (blh - bl02)) + (p10 - bl02);
    //two_sum(&p11, &bl00, p11, bl00);
    bl20=p11;
    p11 = bl20 + bl00;
    bl01 = p11 - bl20;
    bl00 = (bl20 - (p11 - bl01)) + (bl00 - bl01);
    //two_sum(&p10, &p11, p10, p11);
    bl01=p10;
    p10 = bl01 + p11;
    bl20 = p10 - bl01;
    p11 = (bl01 - (p10 - bl20)) + (p11 - bl20);
    //O(eps^4) terms
    al10 = al10 + p20 + bl00 + p11;
    
    //O(eps^3) terms
    bll = p10 + xhh * yll + xhl
            * ylh + xlh * yhl + xll
            * yhh + ah02 + bl11 + al11;
    
    //printf("%e\n",zhl);
    al00=bhh;
    bl00=bhl;
    ah02=blh;
    al11=bll;
    
    //renormalize(&c0[0], &c1[0], &c2[0], &c3[0], al00,bl00,ah02,al11, al10);
    bl02 = al11 + al10;
    bl01 = al10 - (bl02 - al11);
    bh02 = ah02 + bl02;
    bh01 = bl02 - (bh02 - ah02);
    bl02  = bl00 + bh02;
    al01 = bh02 - (bl02 - bl00);
    bhh = al00 + bl02;
    ah01 = bl02 - (bhh - al00);
    bl02 = bh01 + bl01;
    bh01 = bl01 - (bl02 - bh01);
    
    bh02 = al01 + bl02;
    al01 = bl02 - (bh02 - al01);
    
    bhl = ah01 + bh02;
    ah01 = bh02 - (bhl - ah01);
    
    bl02 = al01 + bh01;
    al01 = bh01 - (bl02 -al01);
    
    blh = ah01 + bl02;
    ah01 = bl02 - (blh - ah01);
    bll = ah01 + al01;
    
    *chh = bhh;
    *chl = bhl;
    *clh = blh;
    *cll = bll;
}

//-----------------------------------------------------------------
// FMA
//-----------------------------------------------------------------

/**
 *
 * @fn _d_mul_dd_fma(__m256d*,__m256d*,__m256d,__m256d,__m256d)
 * @brief The function compute the multiplication of double and double-double number with FMA.
 * @param vchi,vclo output (double-double precision)
 * @param vahi input (double precision)
 * @param vbhi,vblo input (double-double precision)
 *
 */
void _d_mul_dd_fma(__m256d* vchi,__m256d* vclo,__m256d vahi,__m256d vbhi,__m256d vblo)
{
    __m256d vp1,vp2,vchi_n,vclo_n;
    //twoprod_fma(ahi, bhi)
    vp1 = _mm256_mul_pd(vahi,vbhi);
    vp2 = _mm256_fmsub_pd(vahi,vbhi,vp1);
    
    vp2 = _mm256_fmadd_pd(vahi , vblo , vp2);
    
    vchi_n =_mm256_add_pd(vp1 , vp2);
    vclo_n = _mm256_sub_pd(vp2, _mm256_sub_pd(vchi_n, vp1));
    
    *vchi = vchi_n;
    *vclo = vclo_n;
}
/**
 *
 * @fn _d_mul_qd_fma(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the multiplication of double and quad-double number with FMA.
 * @param vchh,vchl,vclh,vcll output (quad-double precision)
 * @param vxhh input (double precision)
 * @param vyhh,vyhl,vylh,vyll input (quad-double precision)
 *
 */
void _d_mul_qd_fma(__m256d* vchh,__m256d* vchl,__m256d* vclh,__m256d* vcll,__m256d vxhh,__m256d vyhh,__m256d vyhl,__m256d vylh,__m256d vyll)
{
    __m256d vsh,ve1,ve2,ve3,ve4,veh,vs1,vs2,vs3,vs4,vs5,vp1,vp2,vp3,vp4,vp5,ve,vv,vs,vss,vt1,vt2,vt3,vt4,vzhh,vzhl,vzlh,vzll;
    
    //mul
    //two_prod(a, b.hh);
    vp1 = _mm256_mul_pd(vxhh, vyhh);
    ve1 = _mm256_fmsub_pd(vxhh,vyhh,vp1);
    
    //two_prod(a, b.hl);
    vp2 = _mm256_mul_pd(vxhh,vyhl);
    ve2 = _mm256_fmsub_pd(vxhh,vyhl,vp2);
    
    //two_prod(a, b.lh);
    vp3 = _mm256_mul_pd(vxhh,vylh);
    ve3 = _mm256_fmsub_pd(vxhh,vylh,vp3);
    
    vp4 = _mm256_mul_pd(vxhh,vyll);
    
    //two_Sum(p2, e1);
    vs = _mm256_add_pd(vp2, ve1);
    vv =_mm256_sub_pd(vs, vp2);
    ve1 =_mm256_add_pd(_mm256_sub_pd(vp2 ,_mm256_sub_pd(vs,vv)),_mm256_sub_pd(ve1,vv));
    
    vp2 = vs;
    
    //three_sum(p3, e2, e1);
    //two_Sum(p3, e2);
    vs = _mm256_add_pd(vp3, ve2);
    vv =_mm256_sub_pd(vs, vp3);
    veh =_mm256_add_pd(_mm256_sub_pd(vp3 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve2,vv));
    //two_Sum(s, e1);
    vp3 = _mm256_add_pd(vs, ve1);
    vv =_mm256_sub_pd(vp3, vs);
    vs =_mm256_add_pd(_mm256_sub_pd(vs ,_mm256_sub_pd(vp3,vv)),
            _mm256_sub_pd(ve1,vv));
    //two_Sum(eh, s);
    ve2 = _mm256_add_pd(veh, vs);
    vv =_mm256_sub_pd(ve2, veh);
    ve1 =_mm256_add_pd(_mm256_sub_pd(veh ,_mm256_sub_pd(ve2,vv)),
            _mm256_sub_pd(vs,vv));
    
    
    //three_sum2(p4, e3, e2);
    //two_Sum(p4, e3);
    vs = _mm256_add_pd(vp4, ve3);
    vv =_mm256_sub_pd(vs, vp4);
    veh =_mm256_add_pd(_mm256_sub_pd(vp4 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve3,vv));
    //two_Sum(s, e2);
    vp4 = _mm256_add_pd(vs, ve2);
    vv =_mm256_sub_pd(vp4, vs);
    vs =_mm256_add_pd(_mm256_sub_pd(vs ,_mm256_sub_pd(vp4,vv)),
            _mm256_sub_pd(ve2,vv));
    
    ve2 = _mm256_add_pd(veh, vs);
    
    vp5 = _mm256_add_pd(ve1, ve2);
    
    //renormalize(p1, p2, p3, p4, p5);
    vs = _mm256_add_pd(vp4, vp5);
    vt4 = _mm256_sub_pd(vp5, _mm256_sub_pd(vs, vp4));
    vss = _mm256_add_pd(vp3, vs);
    vt3 = _mm256_sub_pd(vs, _mm256_sub_pd(vss, vp3));
    vs = _mm256_add_pd(vp2, vss);
    vt2 = _mm256_sub_pd(vss, _mm256_sub_pd(vs, vp2));
    vzhh = _mm256_add_pd(vp1, vs);
    vt1 = _mm256_sub_pd(vs, _mm256_sub_pd(vzhh, vp1));
    vs = _mm256_add_pd(vt3, vt4);
    vt3 = _mm256_sub_pd(vt4, _mm256_sub_pd(vs, vt3));
    vss = _mm256_add_pd(vt2, vs);
    vt2 = _mm256_sub_pd(vs, _mm256_sub_pd(vss, vt2));
    vzhl = _mm256_add_pd(vt1, vss);
    vt1 = _mm256_sub_pd(vss, _mm256_sub_pd(vzhl, vt1));
    vs = _mm256_add_pd(vt2, vt3);
    vt2 = _mm256_sub_pd(vt3, _mm256_sub_pd(vs, vt2));
    vzlh = _mm256_add_pd(vt1, vs);
    vt1 = _mm256_sub_pd(vs, _mm256_sub_pd(vzlh, vt1));
    vzll = _mm256_add_pd(vt1, vt2);
    
    *vchh = vzhh;
    *vchl = vzhl;
    *vclh = vzlh;
    *vcll = vzll;
    
}
/**
 *
 * @fn _dd_mul_dd_fma(__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the multiplication of double-double and double-double number with FMA.
 * @param vchi,vclo output (double-double precision)
 * @param vahi input (double precision)
 * @param vbhi,vblo input (quad-double precision)
 *
 */
void _dd_mul_dd_fma(__m256d* vchi,__m256d* vclo,__m256d vahi,__m256d valo,__m256d vbhi,__m256d vblo)
{
    __m256d vp1,vp2,vchi_n,vclo_n;
    
    vp1 = _mm256_mul_pd(vahi,vbhi);
    vp2 = _mm256_fmsub_pd(vahi,vbhi,vp1);
    
    vp2 = _mm256_fmadd_pd(vahi , vblo , vp2);
    vp2 = _mm256_fmadd_pd(valo , vbhi , vp2);
    vchi_n =_mm256_add_pd(vp1 , vp2);
    vclo_n = _mm256_sub_pd(vp2, _mm256_sub_pd(vchi_n, vp1));
    
    *vchi = vchi_n;
    *vclo = vclo_n;
}
/**
 *
 * @fn _dd_mul_qd_fma(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the multiplication of double-double and quad-double number with FMA.
 * @param vchh,vchl,vclh,vcll output (quad-double precision)
 * @param vxhh,vxhl input (double-double precision)
 * @param vyhh,vyhl,vylh,vyll input (quad-double precision)
 *
 */
void _dd_mul_qd_fma(__m256d* vchh,__m256d* vchl,__m256d* vclh,__m256d* vcll,__m256d vxhh,__m256d vxhl,__m256d vyhh,__m256d vyhl,__m256d vylh,__m256d vyll)
{
    __m256d vsh,ve1,ve2,ve3,ve4,ve5,veh,vs1,vs2,vs3,vs4,vs5,vp1,vp2,vp3,vp4,vp5,ve,vv,vs,vss,vt1,vt2,vt3,vt4,vzhh,vzhl,vzlh,vzll;
    //mul
    //two_prod(a.hh, b.hh);
    vp1 = _mm256_mul_pd(vxhh, vyhh);
    ve1 = _mm256_fmsub_pd(vxhh,vyhh,vp1);
    
    //two_prod(a.hl, b.hh);
    vp2 = _mm256_mul_pd(vxhl,vyhh);
    ve2 = _mm256_fmsub_pd(vxhl,vyhh,vp2);
    
    //two_prod(a.hh, b.hl);
    vp3 = _mm256_mul_pd(vxhh,vyhl);
    ve3 = _mm256_fmsub_pd(vxhh,vyhl,vp3);
    
    //two_prod(a.hl, b.hl);
    vp4 = _mm256_mul_pd(vxhl,vyhl);
    ve4 = _mm256_fmsub_pd(vxhl,vyhl,vp4);
    
    //two_prod(a.hh, b.lh);
    vp5 = _mm256_mul_pd(vxhh,vylh);
    ve5 = _mm256_fmsub_pd(vxhh,vylh,vp5);
    
    //three_sum(p2, p3, e1);
    //two_Sum(p2, p3);
    vs = _mm256_add_pd(vp2, vp3);
    vv =_mm256_sub_pd(vs, vp2);
    veh =_mm256_add_pd(_mm256_sub_pd(vp2 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(vp3,vv));
    //two_Sum(s, e1);
    vp2 = _mm256_add_pd(vs, ve1);
    vv =_mm256_sub_pd(vp2, vs);
    vs =_mm256_add_pd(_mm256_sub_pd(vs ,_mm256_sub_pd(vp2,vv)),
            _mm256_sub_pd(ve1,vv));
    //two_Sum(eh, s);
    vp3 = _mm256_add_pd(veh, vs);
    vv =_mm256_sub_pd(vp3, veh);
    ve1 =_mm256_add_pd(_mm256_sub_pd(veh ,_mm256_sub_pd(vp3,vv)),
            _mm256_sub_pd(vs,vv));
    
    //three_sum(p3, p4, p5);
    //two_Sum(p3, p4);
    vs = _mm256_add_pd(vp3, vp4);
    vv =_mm256_sub_pd(vs, vp3);
    veh =_mm256_add_pd(_mm256_sub_pd(vp3 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(vp4,vv));
    //two_Sum(s, p5);
    vp3 = _mm256_add_pd(vs, vp5);
    vv =_mm256_sub_pd(vp3, vs);
    vs =_mm256_add_pd(_mm256_sub_pd(vs ,_mm256_sub_pd(vp3,vv)),
            _mm256_sub_pd(vp5,vv));
    //two_Sum(eh, s);
    vp4 = _mm256_add_pd(veh, vs);
    vv =_mm256_sub_pd(vp4, veh);
    vp5 =_mm256_add_pd(_mm256_sub_pd(veh ,_mm256_sub_pd(vp4,vv)),
            _mm256_sub_pd(vs,vv));
    
    
    //two_Sum(e2, e3);
    vs = _mm256_add_pd(ve2, ve3);
    vv =_mm256_sub_pd(vs, ve2);
    ve3 =_mm256_add_pd(_mm256_sub_pd(ve2 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve3,vv));
    ve2 = vs;
    
    //two_Sum(p3, e2);
    vs = _mm256_add_pd(vp3, ve2);
    vv =_mm256_sub_pd(vs, vp3);
    ve2 =_mm256_add_pd(_mm256_sub_pd(vp3 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve2,vv));
    vp3 = vs;
    
    //two_Sum(p4, e3);
    vs = _mm256_add_pd(vp4, ve3);
    vv =_mm256_sub_pd(vs, vp4);
    ve3 =_mm256_add_pd(_mm256_sub_pd(vp4 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve3,vv));
    vp4 = vs;
    
    //two_Sum(p4, e2);
    vs = _mm256_add_pd(vp4, ve2);
    vv =_mm256_sub_pd(vs, vp4);
    ve2 =_mm256_add_pd(_mm256_sub_pd(vp4 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve2,vv));
    vp4 = vs;
    
    vp5 = _mm256_add_pd(_mm256_add_pd(vp5, ve2),ve3);
    
    vss =_mm256_add_pd(_mm256_fmadd_pd(vxhl,vylh,ve4),_mm256_fmadd_pd(vxhh,vyll,ve5));
    
    //three_sum2(p4, e1, ss);
    //two_Sum(p4, e1);
    vs = _mm256_add_pd(vp4, ve1);
    vv =_mm256_sub_pd(vs, vp4);
    veh =_mm256_add_pd(_mm256_sub_pd(vp4 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve1,vv));
    //two_Sum(s, e2);
    vp4 = _mm256_add_pd(vs, vss);
    vv =_mm256_sub_pd(vp4, vs);
    vs =_mm256_add_pd(_mm256_sub_pd(vs ,_mm256_sub_pd(vp4,vv)),
            _mm256_sub_pd(vss,vv));
    
    ve1 = _mm256_add_pd(veh, vs);
    
    vp5 = _mm256_add_pd(vp5, ve1);
    
    //renormalize(p1, p2, p3, p4, p5);
    vs = _mm256_add_pd(vp4, vp5);
    vt4 = _mm256_sub_pd(vp5, _mm256_sub_pd(vs, vp4));
    vss = _mm256_add_pd(vp3, vs);
    vt3 = _mm256_sub_pd(vs, _mm256_sub_pd(vss, vp3));
    vs = _mm256_add_pd(vp2, vss);
    vt2 = _mm256_sub_pd(vss, _mm256_sub_pd(vs, vp2));
    vzhh = _mm256_add_pd(vp1, vs);
    vt1 = _mm256_sub_pd(vs, _mm256_sub_pd(vzhh, vp1));
    vs = _mm256_add_pd(vt3, vt4);
    vt3 = _mm256_sub_pd(vt4, _mm256_sub_pd(vs, vt3));
    vss = _mm256_add_pd(vt2, vs);
    vt2 = _mm256_sub_pd(vs, _mm256_sub_pd(vss, vt2));
    vzhl = _mm256_add_pd(vt1, vss);
    vt1 = _mm256_sub_pd(vss, _mm256_sub_pd(vzhl, vt1));
    vs = _mm256_add_pd(vt2, vt3);
    vt2 = _mm256_sub_pd(vt3, _mm256_sub_pd(vs, vt2));
    vzlh = _mm256_add_pd(vt1, vs);
    vt1 = _mm256_sub_pd(vs, _mm256_sub_pd(vzlh, vt1));
    vzll = _mm256_add_pd(vt1, vt2);
    
    *vchh = vzhh;
    *vchl = vzhl;
    *vclh = vzlh;
    *vcll = vzll;
}
/**
 *
 * @fn _qd_mul_qd_fma(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the multiplication of quad-double and quad-double number with FMA.
 * @param vchh,vchl,vclh,vcll output (quad-double precision)
 * @param vxhh,vxhl,vxlh,vxll input (quad-double precision)
 * @param vyhh,vyhl,vylh,vyll input (quad-double precision)
 *
 */
void _qd_mul_qd_fma(__m256d* vchh,__m256d* vchl,__m256d* vclh,__m256d* vcll,__m256d vxhh,__m256d vxhl,__m256d vxlh,__m256d vxll,__m256d vyhh,__m256d vyhl,__m256d vylh,__m256d vyll)
{
    __m256d zhh,zhl,zlh,zll;
    __m256d p01,p10,p11,p02,p20,ah00,al00,bh00,bl00,ah01,al01,bh01,bl01,ah10,al10;
    __m256d bh10,bl10,ah11,al11,bh11,bl11,ah20,al20,bh20,bl20,ah02,al02,bh02,bl02;
    //mul
    //two_prod(&c0[0], &al00, a0, b0);
    
    zhh = _mm256_mul_pd(vxhh, vyhh);
    al00 = _mm256_fmsub_pd(vxhh,vyhh,zhh);
    
    //O(eps) terms
    //two_prod(&p01, &bl00, a0, b1);
    p01 = _mm256_mul_pd(vxhh,vyhl);
    bl00 = _mm256_fmsub_pd(vxhh,vyhl,p01);
    
    //two_prod(&p10, &al10, a1, b0);
    p10=_mm256_mul_pd(vxhl,vyhh);
    al10 = _mm256_fmsub_pd(vxhl,vyhh,p10);
    
    //three_sum(&c1[0], &p10, &p01, p01, p10, al00);
    // [ah00, bh00] = Calc.twoSum(p01, p10);
    ah00=_mm256_add_pd(p01,p10);
    al01=_mm256_sub_pd(ah00,p01);
    bh00=_mm256_add_pd(_mm256_sub_pd(p01,_mm256_sub_pd(ah00,al01)),_mm256_sub_pd(p10,al01));
    // [zhl, ah00] = Calc.twoSum(sh, al00);
    zhl=_mm256_add_pd(ah00,al00);
    al01=_mm256_sub_pd(zhl,ah00);
    ah00=_mm256_add_pd(_mm256_sub_pd(ah00,_mm256_sub_pd(zhl,al01)),_mm256_sub_pd(al00,al01));
    // [p10, p01] = Calc.twoSum(eh, sh);
    p10=_mm256_add_pd(bh00,ah00);
    al01=_mm256_sub_pd(p10,bh00);
    p01=_mm256_add_pd(_mm256_sub_pd(bh00,_mm256_sub_pd(p10,al01)),_mm256_sub_pd(ah00,al01));
    //---------------------------------------------------
    //O(eps^2) terms
    //two_prod(&p02, &ah02, a0, b2);
    p02=_mm256_mul_pd(vxhh,vylh);
    ah02 = _mm256_fmsub_pd(vxhh,vylh,p02);
    
    //two_prod(&p11, &bl11, a1, b1);
    p11=_mm256_mul_pd(vxhl,vyhl);
    bl11 =  _mm256_fmsub_pd(vxhl,vyhl,p11);
    
    //two_prod(&p20, &al11, a2, b0);
    p20=_mm256_mul_pd(vxlh,vyhh);
    al11 = _mm256_fmsub_pd(vxlh,vyhh,p20);
    
    //six three sum for p10, bl00, al10, p02, p11, p20
    
    //three_sum(&p10, &bl00, &al10, p10, bl00, al10);
    // [ah20, bh20] = Calc.twoSum(p10, bl00);
    ah20=_mm256_add_pd(p10,bl00);
    al20=_mm256_sub_pd(ah20,p10);
    bh20=_mm256_add_pd(_mm256_sub_pd(p10,_mm256_sub_pd(ah20,al20)),_mm256_sub_pd(bl00,al20));
    // [p10, ah20] = Calc.twoSum(ah20, al10);
    p10=_mm256_add_pd(ah20,al10);
    al20=_mm256_sub_pd(p10,ah20);
    ah20=_mm256_add_pd(_mm256_sub_pd(ah20,_mm256_sub_pd(p10,al20)),_mm256_sub_pd(al10,al20));
    // [bl00, al10] = Calc.twoSum(bh20, ah20);
    bl00=_mm256_add_pd(bh20,ah20);
    al20=_mm256_sub_pd(bl00,bh20);
    al10=_mm256_add_pd(_mm256_sub_pd(bh20,_mm256_sub_pd(bl00,al20)),_mm256_sub_pd(ah20,al20));
    
    
    //three_sum(&p02, &p11, &p20, p02, p11, p20);
    // [ah11, bh02] = Calc.twoSum(p02, p11);
    ah11=_mm256_add_pd(p02,p11);
    al02=_mm256_sub_pd(ah11,p02);
    bh02=_mm256_add_pd(_mm256_sub_pd(p02,_mm256_sub_pd(ah11,al02)),_mm256_sub_pd(p11,al02));
    // [p02, ah11] = Calc.twoSum(ah11, p20);
    p02=_mm256_add_pd(ah11,p20);
    al02=_mm256_sub_pd(p02,ah11);
    ah11= _mm256_add_pd(_mm256_sub_pd(ah11,_mm256_sub_pd(p02,al02)),_mm256_sub_pd(p20,al02));
    // [p11, p20] = Calc.twoSum(bh02, ah11);
    p11=_mm256_add_pd(bh02,bh02);
    al02=_mm256_sub_pd(p11,bh02);
    p20=_mm256_add_pd(_mm256_sub_pd(bh02,_mm256_sub_pd(p11,al02)),_mm256_sub_pd(ah11,al02));
    
    // two_sum(&c2[0], &p10, p02, p10);
    zlh = _mm256_add_pd(p02 , p10);
    bl02 = _mm256_sub_pd(zlh , p02);
    p10 = _mm256_add_pd(_mm256_sub_pd(p02 , _mm256_sub_pd(zlh , bl02)) , _mm256_sub_pd(p10 , bl02));
    //two_sum(&p11, &bl00, p11, bl00);
    bl20=p11;
    p11 = _mm256_add_pd(bl20 , bl00);
    bl01 = _mm256_sub_pd(p11 , bl20);
    bl00 = _mm256_add_pd(_mm256_sub_pd(bl20 , _mm256_sub_pd(p11 , bl01)) , _mm256_sub_pd(bl00 , bl01));
    //two_sum(&p10, &p11, p10, p11);
    bl01=p10;
    p10 = _mm256_add_pd(bl01 , p11);
    bl20 = _mm256_sub_pd(p10 , bl01);
    p11 = _mm256_add_pd(_mm256_sub_pd(bl01 , _mm256_sub_pd(p10 , bl20)) , _mm256_sub_pd(p11 , bl20));
    //O(eps^4) terms
    al10 = _mm256_add_pd(_mm256_add_pd(al10 , p20) , _mm256_add_pd(bl00 , p11));
    
    //no fma
    //O(eps^3) terms
    zll=_mm256_add_pd(
            _mm256_add_pd(
            _mm256_fmadd_pd(vxhh,vyll,
            _mm256_fmadd_pd(
            vxhl,vylh,_mm256_fmadd_pd(
            vxlh,vyhl,_mm256_fmadd_pd(
            vxll,vyhh,ah02)))),
            _mm256_add_pd(bl11,al11)),p10);
    
    al00=zhh;
    bl00=zhl;
    ah02=zlh;
    al11=zll;
    
    //renormalize(&c0[0], &c1[0], &c2[0], &c3[0], al00,bl00,ah02,al11, al10);
    bl02 = _mm256_add_pd(al11 , al10);
    bl01 = _mm256_sub_pd(al10 , _mm256_sub_pd(bl02, al11));
    bh02 = _mm256_add_pd(ah02 , bl02);
    bh01 = _mm256_sub_pd(bl02 , _mm256_sub_pd(bh02 , ah02));
    bl02  = _mm256_add_pd(bl00 , bh02);
    al01 = _mm256_sub_pd(bh02, _mm256_sub_pd(bl02 , bl00));
    zhh = _mm256_add_pd(al00 , bl02);
    ah01 = _mm256_sub_pd(bl02 , _mm256_sub_pd(zhh , al00));
    bl02 = _mm256_add_pd(bh01 , bl01);
    bh01 = _mm256_sub_pd(bl01 ,_mm256_sub_pd(bl02 , bh01));
    bh02 = _mm256_add_pd(al01 , bl02);
    al01 = _mm256_sub_pd(bl02 , _mm256_sub_pd(bh02 ,al01));
    zhl = _mm256_add_pd(ah01 , bh02);
    ah01 = _mm256_sub_pd(bh02 , _mm256_sub_pd(zhl, ah01));
    bl02 = _mm256_add_pd(al01 , bh01);
    al01 = _mm256_sub_pd(bh01 , _mm256_sub_pd(bl02 ,al01));
    zlh = _mm256_add_pd(ah01 , bl02);
    ah01 = _mm256_sub_pd(bl02 , _mm256_sub_pd(zlh , ah01));
    zll = _mm256_add_pd(ah01 , al01);
    
    *vchh = zhh;
    *vchl = zhl;
    *vclh = zlh;
    *vcll = zll;
    
}
//-----------------------------------------------------------------
// AVX2
//-----------------------------------------------------------------

/**
 *
 * @fn _d_add_dd_avx2(__m256d*,__m256d*,__m256d,__m256d,__m256d)
 * @brief The function compute the addition of double and double-double number with AVX2.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (double-double precision)
 *
 */
void _d_add_dd_avx2(__m256d* vchi,__m256d* vclo,__m256d vahi,__m256d vbhi,__m256d vblo)
{
    __m256d vs,vv,veh,vel,vchi_n,vclo_n;
    //threeSum2(ahi, bhi, blo);
    //twosum(ahi,bhi)
    vs = _mm256_add_pd(vahi,vbhi);
    vv = _mm256_sub_pd(vs, vahi);
    veh = _mm256_add_pd(_mm256_sub_pd(vahi ,_mm256_sub_pd(vs, vv)), _mm256_sub_pd(vbhi ,vv));
    //twosum(s,blo)
    vchi_n = _mm256_add_pd(vs , vblo);
    vv = _mm256_sub_pd(vchi_n, vs);
    vel = _mm256_add_pd(_mm256_sub_pd(vs ,_mm256_sub_pd(vchi_n, vv)), _mm256_sub_pd(vblo ,vv));
    vclo_n = _mm256_add_pd(veh, vel);
    
    *vchi = vchi_n;
    *vclo = vclo_n;
}
/**
 *
 * @fn _d_mul_dd_avx2(__m256d*,__m256d*,__m256d,__m256d,__m256d)
 * @brief The function compute the multiplication of double and double-double number with AVX2.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (double-double precision)
 *
 */
void _d_mul_dd_avx2(__m256d* vchi,__m256d* vclo,__m256d vahi,__m256d vbhi,__m256d vblo)
{
    __m256d vt,vah,val,vbh,vbl,vt1,vp1,vp2,vchi_n,vclo_n;
    __m256d vcons=_mm256_set_pd(134217729,134217729,134217729,134217729);
    vt = _mm256_mul_pd(vcons, vahi);
    vah = _mm256_sub_pd(vt ,_mm256_sub_pd(vt, vahi));
    val = _mm256_sub_pd(vahi, vah);
    
    vt1 = _mm256_mul_pd(vcons , vbhi);
    vbh = _mm256_sub_pd(vt1 ,_mm256_sub_pd(vt1, vbhi));
    vbl = _mm256_sub_pd(vbhi ,vbh);
    
    vp1 = _mm256_mul_pd(vahi ,vbhi);
    
    vp2= _mm256_add_pd(
            _mm256_add_pd(
            _mm256_add_pd(
            _mm256_sub_pd(
            _mm256_mul_pd(vah,vbh),vp1),
            _mm256_mul_pd(vah,vbl)),
            _mm256_mul_pd(val,vbh)),
            _mm256_mul_pd(val,vbl));
    
    vp2 = _mm256_add_pd(vp2,_mm256_mul_pd(vahi,vblo));
    
    vchi_n =_mm256_add_pd(vp1 , vp2);
    vclo_n = _mm256_sub_pd(vp2, _mm256_sub_pd(vchi_n, vp1));
    
    *vchi = vchi_n;
    *vclo = vclo_n;
}
/**
 *
 * @fn _d_add_qd_avx2(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the addition of double and quad-double number with AVX2.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param xhh input (double precision)
 * @param yhh,yhl,ylh,yll input (quad-double precision)
 *
 */
void _d_add_qd_avx2(__m256d* vchh,__m256d* vchl,__m256d* vclh,__m256d* vcll,__m256d vxhh,__m256d vyhh,__m256d vyhl,__m256d vylh,__m256d vyll)
{
    __m256d vs1,vs2,vs3,vs4,vs5,ve,vv,vs,vss,vt1,vt2,vt3,vt4,vzhh,vzhl,vzlh,vzll;
    //twoSum(x,y.hh);
    vs1 = _mm256_add_pd(vxhh, vyhh);
    vv = _mm256_sub_pd(vs1, vxhh);
    ve = _mm256_add_pd(
            _mm256_sub_pd(vxhh, _mm256_sub_pd(vs1, vv)),
            _mm256_sub_pd(vyhh, vv));
    
    //twoSum(e,y.hl);
    vs2 = _mm256_add_pd(ve, vyhl);
    vv = _mm256_sub_pd(vs2, ve);
    ve = _mm256_add_pd(
            _mm256_sub_pd(ve, _mm256_sub_pd(vs2, vv)),
            _mm256_sub_pd(vyhl, vv));
    
    //twoSum(e,y.lh);
    vs3 = _mm256_add_pd(ve, vylh);
    vv = _mm256_sub_pd(vs3, ve);
    ve = _mm256_add_pd(
            _mm256_sub_pd(ve, _mm256_sub_pd(vs3, vv)),
            _mm256_sub_pd(vylh, vv));
    
    //twoSum(e,y.ll);
    vs4 = _mm256_add_pd(ve, vyll);
    vv = _mm256_sub_pd(vs4, ve);
    vs5 = _mm256_add_pd(
            _mm256_sub_pd(ve, _mm256_sub_pd(vs4, vv)),
            _mm256_sub_pd(vyll, vv));
    
    //renormalize(s1, s2, s3, s4, s5);
    vs = _mm256_add_pd(vs4, vs5);
    vt4 = _mm256_sub_pd(vs5, _mm256_sub_pd(vs, vs4));
    vss = _mm256_add_pd(vs3, vs);
    vt3 = _mm256_sub_pd(vs, _mm256_sub_pd(vss, vs3));
    vs = _mm256_add_pd(vs2, vss);
    vt2 = _mm256_sub_pd(vss, _mm256_sub_pd(vs, vs2));
    vzhh = _mm256_add_pd(vs1, vs);
    vt1 = _mm256_sub_pd(vs, _mm256_sub_pd(vzhh, vs1));
    vs = _mm256_add_pd(vt3, vt4);
    vt3 = _mm256_sub_pd(vt4, _mm256_sub_pd(vs, vt3));
    vss = _mm256_add_pd(vt2, vs);
    vt2 = _mm256_sub_pd(vs, _mm256_sub_pd(vss, vt2));
    vzhl = _mm256_add_pd(vt1, vss);
    vt1 = _mm256_sub_pd(vss, _mm256_sub_pd(vzhl, vt1));
    vs = _mm256_add_pd(vt2, vt3);
    vt2 = _mm256_sub_pd(vt3, _mm256_sub_pd(vs, vt2));
    vzlh = _mm256_add_pd(vt1, vs);
    vt1 = _mm256_sub_pd(vs, _mm256_sub_pd(vzlh, vt1));
    vzll = _mm256_add_pd(vt1, vt2);
    
    *vchh = vzhh;
    *vchl = vzhl;
    *vclh = vzlh;
    *vcll = vzll;
}
/**
 *
 * @fn _d_mul_qd_avx2(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the multiplication of double and quad-double number with AVX2.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param xhh input (double precision)
 * @param yhh,yhl,ylh,yll input (quad-double precision)
 *
 */
void _d_mul_qd_avx2(__m256d* vchh,__m256d* vchl,__m256d* vclh,__m256d* vcll,__m256d vxhh,__m256d vyhh,__m256d vyhl,__m256d vylh,__m256d vyll)
{
    __m256d vah,val,vbh,vbl,vsh,ve1,ve2,ve3,ve4,veh,vs1,vs2,vs3,vs4,vs5,vp1,vp2,vp3,vp4,vp5,ve,vv,vs,vss,vt1,vt2,vt3,vt4,vzhh,vzhl,vzlh,vzll;
    __m256d vcons=_mm256_set_pd(134217729,134217729,134217729,134217729);
    //mul
    
    //[ah, al] = Calc.split(a);
    vah = _mm256_mul_pd(vcons, vxhh);
    vah = _mm256_sub_pd(vah, _mm256_sub_pd(vah, vxhh));
    val = _mm256_sub_pd(vxhh, vah);
    
    //[bh, bl] = Calc.split(b);
    vbh = _mm256_mul_pd(vcons, vyhh);
    vbh = _mm256_sub_pd(vbh, _mm256_sub_pd(vbh, vyhh));
    vbl = _mm256_sub_pd(vyhh, vbh);
    
    //two_prod(a, b.hh);
    vp1 = _mm256_mul_pd(vxhh, vyhh);
    ve1 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(
            _mm256_mul_pd(vah, vbh), vp1),
            _mm256_mul_pd(vah, vbl)), _mm256_mul_pd(val, vbh)),
            _mm256_mul_pd(val, vbl));
    
    //[bh, bl] = Calc.split(bhl);
    vbh = _mm256_mul_pd(vcons, vyhl);
    vbh = _mm256_sub_pd(vbh, _mm256_sub_pd(vbh, vyhl));
    vbl = _mm256_sub_pd(vyhl, vbh);
    
    //two_prod(a, b.hl);
    vp2 = _mm256_mul_pd(vxhh, vyhl);
    ve2 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(
            _mm256_mul_pd(vah, vbh), vp2),
            _mm256_mul_pd(vah, vbl)), _mm256_mul_pd(val, vbh)),
            _mm256_mul_pd(val, vbl));
    
    
    //[bh, bl] = Calc.split(bhl);
    vbh = _mm256_mul_pd(vcons, vylh);
    vbh = _mm256_sub_pd(vbh, _mm256_sub_pd(vbh, vylh));
    vbl = _mm256_sub_pd(vylh, vbh);
    
    //two_prod(a, b.lh);
    vp3 = _mm256_mul_pd(vxhh, vylh);
    ve3 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(
            _mm256_mul_pd(vah, vbh), vp3),
            _mm256_mul_pd(vah, vbl)), _mm256_mul_pd(val, vbh)),
            _mm256_mul_pd(val, vbl));
    
    vp4 = _mm256_mul_pd(vxhh,vyll);
    
    //two_Sum(p2, e1);
    vs = _mm256_add_pd(vp2, ve1);
    vv =_mm256_sub_pd(vs, vp2);
    ve1 =_mm256_add_pd(_mm256_sub_pd(vp2 ,_mm256_sub_pd(vs,vv)),_mm256_sub_pd(ve1,vv));
    
    vp2 = vs;
    
    //three_sum(p3, e2, e1);
    //two_Sum(p3, e2);
    vs = _mm256_add_pd(vp3, ve2);
    vv =_mm256_sub_pd(vs, vp3);
    veh =_mm256_add_pd(_mm256_sub_pd(vp3 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve2,vv));
    //two_Sum(s, e1);
    vp3 = _mm256_add_pd(vs, ve1);
    vv =_mm256_sub_pd(vp3, vs);
    vs =_mm256_add_pd(_mm256_sub_pd(vs ,_mm256_sub_pd(vp3,vv)),
            _mm256_sub_pd(ve1,vv));
    //two_Sum(eh, s);
    ve2 = _mm256_add_pd(veh, vs);
    vv =_mm256_sub_pd(ve2, veh);
    ve1 =_mm256_add_pd(_mm256_sub_pd(veh ,_mm256_sub_pd(ve2,vv)),
            _mm256_sub_pd(vs,vv));
    
    
    //three_sum2(p4, e3, e2);
    //two_Sum(p4, e3);
    vs = _mm256_add_pd(vp4, ve3);
    vv =_mm256_sub_pd(vs, vp4);
    veh =_mm256_add_pd(_mm256_sub_pd(vp4 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve3,vv));
    //two_Sum(s, e2);
    vp4 = _mm256_add_pd(vs, ve2);
    vv =_mm256_sub_pd(vp4, vs);
    vs =_mm256_add_pd(_mm256_sub_pd(vs ,_mm256_sub_pd(vp4,vv)),
            _mm256_sub_pd(ve2,vv));
    
    ve2 = _mm256_add_pd(veh, vs);
    
    vp5 = _mm256_add_pd(ve1, ve2);
    
    //renormalize(p1, p2, p3, p4, p5);
    vs = _mm256_add_pd(vp4, vp5);
    vt4 = _mm256_sub_pd(vp5, _mm256_sub_pd(vs, vp4));
    vss = _mm256_add_pd(vp3, vs);
    vt3 = _mm256_sub_pd(vs, _mm256_sub_pd(vss, vp3));
    vs = _mm256_add_pd(vp2, vss);
    vt2 = _mm256_sub_pd(vss, _mm256_sub_pd(vs, vp2));
    vzhh = _mm256_add_pd(vp1, vs);
    vt1 = _mm256_sub_pd(vs, _mm256_sub_pd(vzhh, vp1));
    vs = _mm256_add_pd(vt3, vt4);
    vt3 = _mm256_sub_pd(vt4, _mm256_sub_pd(vs, vt3));
    vss = _mm256_add_pd(vt2, vs);
    vt2 = _mm256_sub_pd(vs, _mm256_sub_pd(vss, vt2));
    vzhl = _mm256_add_pd(vt1, vss);
    vt1 = _mm256_sub_pd(vss, _mm256_sub_pd(vzhl, vt1));
    vs = _mm256_add_pd(vt2, vt3);
    vt2 = _mm256_sub_pd(vt3, _mm256_sub_pd(vs, vt2));
    vzlh = _mm256_add_pd(vt1, vs);
    vt1 = _mm256_sub_pd(vs, _mm256_sub_pd(vzlh, vt1));
    vzll = _mm256_add_pd(vt1, vt2);
    
    *vchh = vzhh;
    *vchl = vzhl;
    *vclh = vzlh;
    *vcll = vzll;
}

/**
 *
 * @fn _dd_add_dd_avx2(__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the addition of double-double and double-double number with AVX2.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (quad-double precision)
 *
 */
void _dd_add_dd_avx2(__m256d* vchi,__m256d* vclo,__m256d vahi,__m256d valo,__m256d vbhi,__m256d vblo)
{
    __m256d vsh,vsl,vv,veh,vel,vchi_n,vclo_n;
    vsh = _mm256_add_pd(vahi,vbhi);
    vv = _mm256_sub_pd(vsh, vahi);
    veh = _mm256_add_pd(_mm256_sub_pd(vahi ,_mm256_sub_pd(vsh, vv)), _mm256_sub_pd(vbhi ,vv));
    vsl = _mm256_add_pd(valo , vblo);
    veh = _mm256_add_pd(veh , vsl);
    vchi_n = _mm256_add_pd(vsh, veh);
    vclo_n = _mm256_sub_pd(veh, _mm256_sub_pd(vchi_n ,vsh));
    
    *vchi = vchi_n;
    *vclo = vclo_n;
}
/**
 *
 * @fn _dd_add_dd_avx2_ieee(__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the addition of double-double and double-double number with AVX2.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (quad-double precision)
 *
 */
void _dd_add_dd_avx2_ieee(__m256d* vchi,__m256d* vclo,__m256d vahi,__m256d valo,__m256d vbhi,__m256d vblo)
{
    /* Compute double-double = double-double + double-double */
    __m256d vs1,vs2,vbv,vt1,vt2,vchi_n,vclo_n;
    /* Add two high-order words. */
    vs1 = _mm256_add_pd(vahi, vbhi);
    vbv = _mm256_sub_pd(vs1, vahi);
    vs2 = _mm256_add_pd(_mm256_sub_pd(vahi ,_mm256_sub_pd(vs1, vbv)), _mm256_sub_pd(vbhi ,vbv));
    /* Add two low-order words. */
    vt1 = _mm256_add_pd(valo, vblo);
    vbv = _mm256_sub_pd(vt1, valo);
    vt2 = _mm256_add_pd(_mm256_sub_pd(valo ,_mm256_sub_pd(vt1, vbv)), _mm256_sub_pd(vblo ,vbv));
    vs2 = _mm256_add_pd(vs2, vt1);
    /* Renormalize (s1, s2) to (t1, s2) */
    vt1 = _mm256_add_pd(vs1, vs2);
    vs2 = _mm256_sub_pd(vs2, _mm256_sub_pd(vt1, vs1));
    vt2 = _mm256_add_pd(vt2, vs2);
    /* Renormalize (t1, t2) */
    vchi_n = _mm256_add_pd(vt1, vt2);
    vclo_n = _mm256_sub_pd(vt2, _mm256_sub_pd(vchi_n, vt1));
    
    *vchi = vchi_n;
    *vclo = vclo_n;
}
/**
 *
 * @fn _dd_mul_dd_avx2(__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the multiplication of double-double and double-double number with AVX2.
 * @param chi,clo output (double-double precision)
 * @param ahi input (double precision)
 * @param bhi,blo input (quad-double precision)
 *
 */
void _dd_mul_dd_avx2(__m256d* vchi,__m256d* vclo,__m256d vahi,__m256d valo,__m256d vbhi,__m256d vblo)
{
    __m256d vt,vt1,vah,val,vbh,vbl,vp1,vp2,vchi_n,vclo_n;
    __m256d vcons=_mm256_set_pd(134217729,134217729,134217729,134217729);
    
    vt = _mm256_mul_pd(vcons, vahi);
    vah = _mm256_sub_pd(vt ,_mm256_sub_pd(vt, vahi));
    val = _mm256_sub_pd(vahi, vah);
    
    vt1 = _mm256_mul_pd(vcons , vbhi);
    vbh = _mm256_sub_pd(vt1 ,_mm256_sub_pd(vt1, vbhi));
    vbl = _mm256_sub_pd(vbhi ,vbh);
    
    vp1 = _mm256_mul_pd(vahi ,vbhi);
    vp2= _mm256_add_pd(
            _mm256_add_pd(
            _mm256_add_pd(
            _mm256_sub_pd(
            _mm256_mul_pd(vah,vbh),vp1),
            _mm256_mul_pd(vah,vbl)),
            _mm256_mul_pd(val,vbh)),
            _mm256_mul_pd(val,vbl));
    
    vp2 = _mm256_add_pd(vp2,_mm256_mul_pd(vahi,vblo));
    vp2 = _mm256_add_pd(vp2,_mm256_mul_pd(valo,vbhi));
    vchi_n =_mm256_add_pd(vp1 , vp2);
    vclo_n = _mm256_sub_pd(vp2, _mm256_sub_pd(vchi_n, vp1));
    
    *vchi = vchi_n;
    *vclo = vclo_n;
}
/**
 *
 * @fn _dd_add_qd_avx2(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the addition of double-double and quad-double number with AVX2.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param xhh,xhl input (double-double precision)
 * @param yhh,yhl,ylh,yll input (quad-double precision)
 *
 */
void _dd_add_qd_avx2(__m256d* vchh,__m256d* vchl,__m256d* vclh,__m256d* vcll,__m256d vxhh,__m256d vxhl,__m256d vyhh,__m256d vyhl,__m256d vylh,__m256d vyll)
{
    __m256d vs1,vs2,vs3,vs4,vs5,ve1,vu,ve2,ve3,ve4,ve5,vv,vs,vss,vt1,vt2,vt3,vt4,vzhh,vzhl,vzlh,vzll;
    
    //two_sumtwoSum(x.hh,y.hh);
    vs1 = _mm256_add_pd(vxhh, vyhh);
    vv = _mm256_sub_pd(vs1, vxhh);
    ve1 = _mm256_add_pd(
            _mm256_sub_pd(vxhh, _mm256_sub_pd(vs1, vv)),
            _mm256_sub_pd(vyhh, vv));
    
    //two_sumtwoSum(x.hl,y.hl);
    vt1 = _mm256_add_pd(vxhl, vyhl);
    vv = _mm256_sub_pd(vt1, vxhl);
    ve2 = _mm256_add_pd(
            _mm256_sub_pd(vxhl, _mm256_sub_pd(vt1, vv)),
            _mm256_sub_pd(vyhl, vv));
    
    //two_sumtwoSum(t1, e1);
    vs2 = _mm256_add_pd(vt1, ve1);
    vv = _mm256_sub_pd(vs2, vt1);
    vu = _mm256_add_pd(
            _mm256_sub_pd(vt1, _mm256_sub_pd(vs2, vv)),
            _mm256_sub_pd(ve1, vv));
    
    //two_sumtwoSum(e2, y.lh);
    vt2 = _mm256_add_pd(ve2, vylh);
    vv = _mm256_sub_pd(vt2, ve2);
    ve3 = _mm256_add_pd(
            _mm256_sub_pd(ve2, _mm256_sub_pd(vt2, vv)),
            _mm256_sub_pd(vylh, vv));
    
    //two_sumtwoSum(u, t2);
    vs3 = _mm256_add_pd(vu, vt2);
    vv = _mm256_sub_pd(vs3, vu);
    vu = _mm256_add_pd(
            _mm256_sub_pd(vu, _mm256_sub_pd(vs3, vv)),
            _mm256_sub_pd(vt2, vv));
    
    //two_sumtwoSum(u, e3);
    vt3 = _mm256_add_pd(vu, ve3);
    vv = _mm256_sub_pd(vt3, vu);
    ve4 = _mm256_add_pd(
            _mm256_sub_pd(vu, _mm256_sub_pd(vt3, vv)),
            _mm256_sub_pd(ve3, vv));
    
    //two_sumtwoSum(t3, y.ll);
    vs4 = _mm256_add_pd(vt3, vyll);
    vv = _mm256_sub_pd(vs4, vt3);
    ve5 = _mm256_add_pd(
            _mm256_sub_pd(vt3, _mm256_sub_pd(vs4, vv)),
            _mm256_sub_pd(vyll, vv));
    
    vs5 = _mm256_add_pd(ve4, ve5);
    
    //renormalize(s1, s2, s3, s4, s5);
    vs = _mm256_add_pd(vs4, vs5);
    vt4 = _mm256_sub_pd(vs5, _mm256_sub_pd(vs, vs4));
    vss = _mm256_add_pd(vs3, vs);
    vt3 = _mm256_sub_pd(vs, _mm256_sub_pd(vss, vs3));
    vs = _mm256_add_pd(vs2, vss);
    vt2 = _mm256_sub_pd(vss, _mm256_sub_pd(vs, vs2));
    vzhh = _mm256_add_pd(vs1, vs);
    vt1 = _mm256_sub_pd(vs, _mm256_sub_pd(vzhh, vs1));
    vs = _mm256_add_pd(vt3, vt4);
    vt3 = _mm256_sub_pd(vt4, _mm256_sub_pd(vs, vt3));
    vss = _mm256_add_pd(vt2, vs);
    vt2 = _mm256_sub_pd(vs, _mm256_sub_pd(vss, vt2));
    vzhl = _mm256_add_pd(vt1, vss);
    vt1 = _mm256_sub_pd(vss, _mm256_sub_pd(vzhl, vt1));
    vs = _mm256_add_pd(vt2, vt3);
    vt2 = _mm256_sub_pd(vt3, _mm256_sub_pd(vs, vt2));
    vzlh = _mm256_add_pd(vt1, vs);
    vt1 = _mm256_sub_pd(vs, _mm256_sub_pd(vzlh, vt1));
    vzll = _mm256_add_pd(vt1, vt2);
    
    *vchh = vzhh;
    *vchl = vzhl;
    *vclh = vzlh;
    *vcll = vzll;
}
/**
 *
 * @fn _dd_mul_qd_avx2(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the multiplication of double-double and quad-double number with AVX2.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param xhh,xhl input (double-double precision)
 * @param yhh,yhl,ylh,yll input (quad-double precision)
 *
 */
void _dd_mul_qd_avx2(__m256d* vchh,__m256d* vchl,__m256d* vclh,__m256d* vcll,__m256d vxhh,__m256d vxhl,__m256d vyhh,__m256d vyhl,__m256d vylh,__m256d vyll)
{
    __m256d va1h,va1l,vbh,vbl,va2h,va2l,vsh,ve1,ve2,ve3,ve4,ve5,veh,vs1,vs2,vs3,vs4,vs5,vp1,vp2,vp3,vp4,vp5,ve,vv,vs,vss,vt1,vt2,vt3,vt4,vzhh,vzhl,vzlh,vzll;
    __m256d vcons=_mm256_set_pd(134217729,134217729,134217729,134217729);
    //mul
    //two_prod(a.hh, b.hh);
    vp1 = _mm256_mul_pd(vxhh, vyhh);
    //split(a)
    va1h = _mm256_mul_pd(vcons, vxhh);
    va1h = _mm256_sub_pd(va1h ,_mm256_sub_pd(va1h, vxhh));
    va1l = _mm256_sub_pd(vxhh, va1h);
    //split(b)
    vbh = _mm256_mul_pd(vcons, vyhh);
    vbh = _mm256_sub_pd(vbh ,_mm256_sub_pd(vbh, vyhh));
    vbl = _mm256_sub_pd(vyhh, vbh);
    ve1 = _mm256_add_pd(
            _mm256_add_pd(
            _mm256_add_pd(
            _mm256_sub_pd(
            _mm256_mul_pd(va1h,vbh),vp1),
            _mm256_mul_pd(va1h,vbl)),
            _mm256_mul_pd(va1l,vbh)),
            _mm256_mul_pd(va1l,vbl));
    
    //two_prod(a.hl, b.hh);
    vp2 = _mm256_mul_pd(vxhl,vyhh);
    //split(a)
    va2h = _mm256_mul_pd(vcons, vxhl);
    va2h = _mm256_sub_pd(va2h ,_mm256_sub_pd(va2h, vxhl));
    va2l = _mm256_sub_pd(vxhl, va2h);
    ve2 = _mm256_add_pd(
            _mm256_add_pd(
            _mm256_add_pd(
            _mm256_sub_pd(
            _mm256_mul_pd(va2h,vbh),vp2),
            _mm256_mul_pd(va2h,vbl)),
            _mm256_mul_pd(va2l,vbh)),
            _mm256_mul_pd(va2l,vbl));
    
    //two_prod(a.hh, b.hl);
    vp3 = _mm256_mul_pd(vxhh,vyhl);
    //split(b)
    vbh = _mm256_mul_pd(vcons, vyhl);
    vbh = _mm256_sub_pd(vbh ,_mm256_sub_pd(vbh, vyhl));
    vbl = _mm256_sub_pd(vyhl, vbh);
    ve3 = _mm256_add_pd(
            _mm256_add_pd(
            _mm256_add_pd(
            _mm256_sub_pd(
            _mm256_mul_pd(va1h,vbh),vp3),
            _mm256_mul_pd(va1h,vbl)),
            _mm256_mul_pd(va1l,vbh)),
            _mm256_mul_pd(va1l,vbl));
    
    //two_prod(a.hl, b.hl);
    vp4 = _mm256_mul_pd(vxhl,vyhl);
    ve4 = _mm256_add_pd(
            _mm256_add_pd(
            _mm256_add_pd(
            _mm256_sub_pd(
            _mm256_mul_pd(va2h,vbh),vp4),
            _mm256_mul_pd(va2h,vbl)),
            _mm256_mul_pd(va2l,vbh)),
            _mm256_mul_pd(va2l,vbl));
    
    //two_prod(a.hh, b.lh);
    vp5 = _mm256_mul_pd(vxhh,vylh);
    //split(b)
    vbh = _mm256_mul_pd(vcons, vylh);
    vbh = _mm256_sub_pd(vbh ,_mm256_sub_pd(vbh, vylh));
    vbl = _mm256_sub_pd(vylh, vbh);
    ve5 = _mm256_add_pd(
            _mm256_add_pd(
            _mm256_add_pd(
            _mm256_sub_pd(
            _mm256_mul_pd(va1h,vbh),vp5),
            _mm256_mul_pd(va1h,vbl)),
            _mm256_mul_pd(va1l,vbh)),
            _mm256_mul_pd(va1l,vbl));
    
    //three_sum(p2, p3, e1);
    //two_Sum(p2, p3);
    vs = _mm256_add_pd(vp2, vp3);
    vv =_mm256_sub_pd(vs, vp2);
    veh =_mm256_add_pd(_mm256_sub_pd(vp2 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(vp3,vv));
    //two_Sum(s, e1);
    vp2 = _mm256_add_pd(vs, ve1);
    vv =_mm256_sub_pd(vp2, vs);
    vs =_mm256_add_pd(_mm256_sub_pd(vs ,_mm256_sub_pd(vp2,vv)),
            _mm256_sub_pd(ve1,vv));
    //two_Sum(eh, s);
    vp3 = _mm256_add_pd(veh, vs);
    vv =_mm256_sub_pd(vp3, veh);
    ve1 =_mm256_add_pd(_mm256_sub_pd(veh ,_mm256_sub_pd(vp3,vv)),
            _mm256_sub_pd(vs,vv));
    
    //three_sum(p3, p4, p5);
    //two_Sum(p3, p4);
    vs = _mm256_add_pd(vp3, vp4);
    vv =_mm256_sub_pd(vs, vp3);
    veh =_mm256_add_pd(_mm256_sub_pd(vp3 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(vp4,vv));
    //two_Sum(s, p5);
    vp3 = _mm256_add_pd(vs, vp5);
    vv =_mm256_sub_pd(vp3, vs);
    vs =_mm256_add_pd(_mm256_sub_pd(vs ,_mm256_sub_pd(vp3,vv)),
            _mm256_sub_pd(vp5,vv));
    //two_Sum(eh, s);
    vp4 = _mm256_add_pd(veh, vs);
    vv =_mm256_sub_pd(vp4, veh);
    vp5 =_mm256_add_pd(_mm256_sub_pd(veh ,_mm256_sub_pd(vp4,vv)),
            _mm256_sub_pd(vs,vv));
    
    
    //two_Sum(e2, e3);
    vs = _mm256_add_pd(ve2, ve3);
    vv =_mm256_sub_pd(vs, ve2);
    ve3 =_mm256_add_pd(_mm256_sub_pd(ve2 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve3,vv));
    ve2 = vs;
    
    //two_Sum(p3, e2);
    vs = _mm256_add_pd(vp3, ve2);
    vv =_mm256_sub_pd(vs, vp3);
    ve2 =_mm256_add_pd(_mm256_sub_pd(vp3 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve2,vv));
    vp3 = vs;
    
    //two_Sum(p4, e3);
    vs = _mm256_add_pd(vp4, ve3);
    vv =_mm256_sub_pd(vs, vp4);
    ve3 =_mm256_add_pd(_mm256_sub_pd(vp4 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve3,vv));
    vp4 = vs;
    
    //two_Sum(p4, e2);
    vs = _mm256_add_pd(vp4, ve2);
    vv =_mm256_sub_pd(vs, vp4);
    ve2 =_mm256_add_pd(_mm256_sub_pd(vp4 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve2,vv));
    vp4 = vs;
    
    vp5 = _mm256_add_pd(_mm256_add_pd(vp5, ve2),ve3);
    
    vss =_mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(vxhl,vylh),ve4),_mm256_add_pd(_mm256_mul_pd(vxhh,vyll),ve5));
    
    //three_sum2(p4, e1, ss);
    //two_Sum(p4, e1);
    vs = _mm256_add_pd(vp4, ve1);
    vv =_mm256_sub_pd(vs, vp4);
    veh =_mm256_add_pd(_mm256_sub_pd(vp4 ,_mm256_sub_pd(vs,vv)),
            _mm256_sub_pd(ve1,vv));
    //two_Sum(s, e2);
    vp4 = _mm256_add_pd(vs, vss);
    vv =_mm256_sub_pd(vp4, vs);
    vs =_mm256_add_pd(_mm256_sub_pd(vs ,_mm256_sub_pd(vp4,vv)),
            _mm256_sub_pd(vss,vv));
    
    ve1 = _mm256_add_pd(veh, vs);
    
    vp5 = _mm256_add_pd(vp5, ve1);
    
    //renormalize(p1, p2, p3, p4, p5);
    vs = _mm256_add_pd(vp4, vp5);
    vt4 = _mm256_sub_pd(vp5, _mm256_sub_pd(vs, vp4));
    vss = _mm256_add_pd(vp3, vs);
    vt3 = _mm256_sub_pd(vs, _mm256_sub_pd(vss, vp3));
    vs = _mm256_add_pd(vp2, vss);
    vt2 = _mm256_sub_pd(vss, _mm256_sub_pd(vs, vp2));
    vzhh = _mm256_add_pd(vp1, vs);
    vt1 = _mm256_sub_pd(vs, _mm256_sub_pd(vzhh, vp1));
    vs = _mm256_add_pd(vt3, vt4);
    vt3 = _mm256_sub_pd(vt4, _mm256_sub_pd(vs, vt3));
    vss = _mm256_add_pd(vt2, vs);
    vt2 = _mm256_sub_pd(vs, _mm256_sub_pd(vss, vt2));
    vzhl = _mm256_add_pd(vt1, vss);
    vt1 = _mm256_sub_pd(vss, _mm256_sub_pd(vzhl, vt1));
    vs = _mm256_add_pd(vt2, vt3);
    vt2 = _mm256_sub_pd(vt3, _mm256_sub_pd(vs, vt2));
    vzlh = _mm256_add_pd(vt1, vs);
    vt1 = _mm256_sub_pd(vs, _mm256_sub_pd(vzlh, vt1));
    vzll = _mm256_add_pd(vt1, vt2);
    
    *vchh = vzhh;
    *vchl = vzhl;
    *vclh = vzlh;
    *vcll = vzll;
}
/**
 *
 * @fn _qd_add_qd_avx2(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the addition of quad-double and quad-double number with AVX2.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param xhh,xhl,xlh,xll input (quad-double precision)
 * @param yhh,yhl,ylh,yll input (quad-double precision)
 *
 */
void _qd_add_qd_avx2(__m256d* vzhh,__m256d* vzhl,__m256d* vzlh,__m256d* vzll,__m256d vxhh,__m256d vxhl,__m256d vxlh,__m256d vxll,__m256d vyhh,__m256d vyhl,__m256d vylh,__m256d vyll)
{
    __m256d vchh,vchl,vclh,vcll,e1,e2,e3,e4,v,sh,eh,t1,t2,t3,t4;
    
    //two_sum(&c0, &e1, a0, b0);
    vchh = _mm256_add_pd(vxhh, vyhh);
    v = _mm256_sub_pd(vchh, vxhh);
    e1 = _mm256_add_pd(_mm256_sub_pd(vxhh, _mm256_sub_pd(vchh, v)),
            _mm256_sub_pd(vyhh, v));
    
    //two_sum(&c1, &e2, a1, b1);
    vchl = _mm256_add_pd(vxhl, vyhl);
    v = _mm256_sub_pd(vchl, vxhl);
    e2 = _mm256_add_pd(_mm256_sub_pd(vxhl, _mm256_sub_pd(vchl, v)),
            _mm256_sub_pd(vyhl, v));
    
    //two_sum(&c1, &e1, c1, e1);
    e3 = _mm256_add_pd(vchl, e1);
    v = _mm256_sub_pd(e3, vchl);
    // check!!!
    e1 = _mm256_add_pd(_mm256_sub_pd(vchl, _mm256_sub_pd(e3, v)), _mm256_sub_pd(e1, v));
    vchl=e3;
    
    //two_sum(&c2, &e3, a2, b2);
    vclh = _mm256_add_pd(vxlh, vylh);
    v = _mm256_sub_pd(vclh, vxlh);
    e3 = _mm256_add_pd(_mm256_sub_pd(vxlh, _mm256_sub_pd(vclh, v)), _mm256_sub_pd(vylh, v));
    //three_sum(&c2, &e1, &e2, c2, e2, e1);
    sh = _mm256_add_pd(vclh, e2);
    v = _mm256_sub_pd(sh, vclh);
    eh = _mm256_add_pd(_mm256_sub_pd(vclh, _mm256_sub_pd(sh, v)),
            _mm256_sub_pd(e2, v));
    
    vclh = _mm256_add_pd(sh, e1);
    v = _mm256_sub_pd(vclh, sh);
    sh = _mm256_add_pd(_mm256_sub_pd(sh,
            _mm256_sub_pd(vclh, v)), _mm256_sub_pd(e1, v));
    
    e1 = _mm256_add_pd(eh, sh);
    v = _mm256_sub_pd(e1, eh);
    e2 = _mm256_add_pd(_mm256_sub_pd(eh, _mm256_sub_pd(e1, v)), _mm256_sub_pd(sh, v));
    
    //two_sum(&c3, &e4, a3, b3);
    vcll = _mm256_add_pd(vxll, vyll);
    v = _mm256_sub_pd(vcll, vxll);
    e4 = _mm256_add_pd(_mm256_sub_pd(vxll, _mm256_sub_pd(vcll, v)), _mm256_sub_pd(vyll, v));
    
    //three_sum2(&c3, &e1, c3, e3, e1);
    sh = _mm256_add_pd(vcll, e3);
    v = _mm256_sub_pd(sh, vcll);
    eh = _mm256_add_pd(_mm256_sub_pd(vcll,
            _mm256_sub_pd(sh, v)), _mm256_sub_pd(e3, v));
    
    vcll = _mm256_add_pd(sh, e1);
    v = _mm256_sub_pd(vcll, sh);
    
    e1 = _mm256_add_pd(eh,
            _mm256_add_pd(_mm256_sub_pd(sh,
            _mm256_sub_pd(vcll,v)),
            _mm256_sub_pd(e1,v)));
    
    e1 = _mm256_add_pd(_mm256_add_pd(e1, e2), e4);
    eh=vchh;
    e2=vchl;
    e3=vclh;
    e4=vcll;
    
    //        renormalize(&c0, &c1, &c2, &c3, c0, c1, c2, c3, e1);
    sh = _mm256_add_pd(e4, e1);
    t4 = _mm256_sub_pd(e1, _mm256_sub_pd(sh, e4));
    v = _mm256_add_pd(e3, sh);
    t3 = _mm256_sub_pd(sh, _mm256_sub_pd(v, e3));
    sh = _mm256_add_pd(e2, v);
    t2 = _mm256_sub_pd(v, _mm256_sub_pd(sh, e2));
    vchh = _mm256_add_pd(eh, sh);
    t1 = _mm256_sub_pd(sh, _mm256_sub_pd(vchh, eh));
    sh = _mm256_add_pd(t3, t4);
    t3 = _mm256_sub_pd(t4, _mm256_sub_pd(sh, t3));
    v = _mm256_add_pd(t2, sh);
    t2 = _mm256_sub_pd(sh, _mm256_sub_pd(v, t2));
    vchl = _mm256_add_pd(t1, v);
    t1 = _mm256_sub_pd(v, _mm256_sub_pd(vchl, t1));
    sh = _mm256_add_pd(t2, t3);
    t2 = _mm256_sub_pd(t3, _mm256_sub_pd(sh, t2));
    vclh = _mm256_add_pd(t1, sh);
    t1 = _mm256_sub_pd(sh, _mm256_sub_pd(vclh, t1));
    vcll = _mm256_add_pd(t1, t2);
    
    *vzhh = vchh;
    *vzhl = vchl;
    *vzlh = vclh;
    *vzll = vcll;
}
/**
 *
 * @fn _qd_mul_qd_avx2(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d)
 * @brief The function compute the multiplication of quad-double and quad-double number with AVX2.
 * @param chh,chl,clh,cll output (quad-double precision)
 * @param xhh,xhl,xlh,xll input (quad-double precision)
 * @param yhh,yhl,ylh,yll input (quad-double precision)
 *
 */
void _qd_mul_qd_avx2(__m256d* vchh,__m256d* vchl,__m256d* vclh,__m256d* vcll,__m256d vxhh,__m256d vxhl,__m256d vxlh,__m256d vxll,__m256d vyhh,__m256d vyhl,__m256d vylh,__m256d vyll)
{
    __m256d vzhh,vzhl,vzlh,vzll,vp01,vp10,vp11,vp02,vp20,vah00,val00,vbh00,vbl00,vah01,val01,vbh01,vbl01,vah10,val10, vbh10,vbl10,vah11,val11,vbh11,vbl11,vah20,val20,vbh20,vbl20,vah02,val02,vbh02,vbl02;
    __m256d vcons=_mm256_set_pd(134217729,134217729,134217729,134217729);
    //two_prod(&c0[0], &al00, a0, b0);
    
    vzhh = _mm256_mul_pd(vxhh, vyhh);
    //[ah, al] = Calc.split(a);
    vah00 = _mm256_mul_pd(vcons, vxhh);
    vah00 = _mm256_sub_pd(vah00, _mm256_sub_pd(vah00, vxhh));
    val00 = _mm256_sub_pd(vxhh, vah00);
    //[bh, bl] = Calc.split(b);
    vbh00 = _mm256_mul_pd(vcons, vyhh);
    vbh00 = _mm256_sub_pd(vbh00, _mm256_sub_pd(vbh00, vyhh));
    vbl00 = _mm256_sub_pd(vyhh, vbh00);
    // no fma
    val00 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(
            _mm256_mul_pd(vah00, vbh00), vzhh),
            _mm256_mul_pd(vah00, vbl00)), _mm256_mul_pd(val00, vbh00)),
            _mm256_mul_pd(val00, vbl00));
    //O(eps) terms
    //two_prod(&p01, &bl00, a0, b1);
    vp01=_mm256_mul_pd(vxhh,vyhl);
    //split
    vah01= _mm256_mul_pd(vcons, vxhh);
    vah01=_mm256_sub_pd(vah01,_mm256_sub_pd(vah01,vxhh));
    val01=_mm256_sub_pd(vxhh,vah01);
    //split
    vbh01= _mm256_mul_pd(vcons, vyhl);
    vbh01=_mm256_sub_pd(vbh01,_mm256_sub_pd(vbh01,vyhl));
    vbl01=_mm256_sub_pd(vyhl,vbh01);
    //no fma
    vbl00=_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(
            _mm256_mul_pd(vah01, vbh01), vp01),
            _mm256_mul_pd(vah01, vbl01)), _mm256_mul_pd(val01, vbh01)),
            _mm256_mul_pd(val01, vbl01));
    
    //two_prod(&p10, &al10, a1, b0);
    vp10=_mm256_mul_pd(vxhl,vyhh);
    //split
    vah10= _mm256_mul_pd(vcons, vxhl);
    vah10=_mm256_sub_pd(vah10,_mm256_sub_pd(vah10,vxhl));
    val10=_mm256_sub_pd(vxhl,vah10);
    //split
    vbh10= _mm256_mul_pd(vcons, vyhh);
    vbh10= _mm256_sub_pd(vbh10,_mm256_sub_pd(vbh10,vyhh));
    vbl10= _mm256_sub_pd(vyhh,vbh10);
    //no fma
    val10=_mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(
            _mm256_mul_pd(vah10, vbh10), vp10),
            _mm256_mul_pd(vah10, vbl10)), _mm256_mul_pd(val10, vbh10)),
            _mm256_mul_pd(val10, vbl10));
    
    
    //three_sum(&c1[0], &p10, &p01, p01, p10, al00);
    // [ah00, bh00] = Calc.twoSum(p01, p10);
    vah00=_mm256_add_pd(vp01,vp10);
    val01=_mm256_sub_pd(vah00,vp01);
    vbh00=_mm256_add_pd(_mm256_sub_pd(vp01,_mm256_sub_pd(vah00,val01)),_mm256_sub_pd(vp10,val01));
    // [zhl, ah00] = Calc.twoSum(sh, al00);
    vzhl=_mm256_add_pd(vah00,val00);
    val01=_mm256_sub_pd(vzhl,vah00);
    vah00=_mm256_add_pd(_mm256_sub_pd(vah00,_mm256_sub_pd(vzhl,val01)),_mm256_sub_pd(val00,val01));
    // [p10, p01] = Calc.twoSum(eh, sh);
    vp10=_mm256_add_pd(vbh00,vah00);
    val01=_mm256_sub_pd(vp10,vbh00);
    vp01=_mm256_add_pd(_mm256_sub_pd(vbh00,_mm256_sub_pd(vp10,val01)),_mm256_sub_pd(vah00,val01));
    //---------------------------------------------------
    //O(eps^2) terms
    //two_prod(&p02, &ah02, a0, b2);
    vp02=_mm256_mul_pd(vxhh,vylh);
    //split
    vah02= _mm256_mul_pd(vcons, vxhh);
    vah02=_mm256_sub_pd(vah02,_mm256_sub_pd(vah02,vxhh));
    val02=_mm256_sub_pd(vxhh,vah02);
    //split
    vbh02= _mm256_mul_pd(vcons, vylh);
    vbh02=_mm256_sub_pd(vbh02,_mm256_sub_pd(vbh02,vylh));
    vbl02=_mm256_sub_pd(vylh,vbh02);
    
    // no fma
    vah02 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(
            _mm256_mul_pd(vah02, vbh02), vp02),
            _mm256_mul_pd(vah02, vbl02)), _mm256_mul_pd(val02, vbh02)),
            _mm256_mul_pd(val02, vbl02));
    
    //two_prod(&p11, &bl11, a1, b1);
    vp11=_mm256_mul_pd(vxhl,vyhl);
    //split
    vah11= _mm256_mul_pd(vcons, vxhl);
    vah11=_mm256_sub_pd(vah11,_mm256_sub_pd(vah11,vxhl));
    val11=_mm256_sub_pd(vxhl, vah11);
    //split
    vbh11= _mm256_mul_pd(vcons, vyhl);
    vbh11=_mm256_sub_pd(vbh11,_mm256_sub_pd(vbh11,vyhl));
    vbl11=_mm256_sub_pd(vyhl,vbh11);
    
    // no fma
    vbl11 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(
            _mm256_mul_pd(vah11, vbh11), vp11),
            _mm256_mul_pd(vah11, vbl11)), _mm256_mul_pd(val11, vbh11)),
            _mm256_mul_pd(val11, vbl11));
    
    //two_prod(&p20, &al11, a2, b0);
    vp20=_mm256_mul_pd(vxlh,vyhh);
    //split
    vah20= _mm256_mul_pd(vcons, vxlh);
    vah20=_mm256_sub_pd(vah20,_mm256_sub_pd(vah20,vxlh));
    val20=_mm256_sub_pd(vxlh,vah20);
    //split
    vbh20=_mm256_mul_pd(vcons, vyhh);
    vbh20=_mm256_sub_pd(vbh20, _mm256_sub_pd(vbh20, vyhh));
    vbl20=_mm256_sub_pd(vyhh, vbh20);
    
    // no fma
    val11 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_sub_pd(
            _mm256_mul_pd(vah20, vbh20), vp20),
            _mm256_mul_pd(vah20, vbl20)), _mm256_mul_pd(val20, vbh20)),
            _mm256_mul_pd(val20, vbl20));
    
    //six three sum for p10, bl00, al10, p02, p11, p20
    
    //three_sum(&p10, &bl00, &al10, p10, bl00, al10);
    // [ah20, bh20] = Calc.twoSum(p10, bl00);
    vah20=_mm256_add_pd(vp10,vbl00);
    val20=_mm256_sub_pd(vah20,vp10);
    vbh20=_mm256_add_pd(_mm256_sub_pd(vp10,_mm256_sub_pd(vah20,val20)),_mm256_sub_pd(vbl00,val20));
    // [p10, ah20] = Calc.twoSum(ah20, al10);
    vp10=_mm256_add_pd(vah20,val10);
    val20=_mm256_sub_pd(vp10,vah20);
    vah20=_mm256_add_pd(_mm256_sub_pd(vah20,_mm256_sub_pd(vp10,val20)),_mm256_sub_pd(val10,val20));
    // [bl00, al10] = Calc.twoSum(bh20, ah20);
    vbl00=_mm256_add_pd(vbh20,vah20);
    val20=_mm256_sub_pd(vbl00,vbh20);
    val10=_mm256_add_pd(_mm256_sub_pd(vbh20,_mm256_sub_pd(vbl00,val20)),_mm256_sub_pd(vah20,val20));
    
    
    //three_sum(&p02, &p11, &p20, p02, p11, p20);
    // [ah11, bh02] = Calc.twoSum(p02, p11);
    vah11=_mm256_add_pd(vp02,vp11);
    val02=_mm256_sub_pd(vah11,vp02);
    vbh02=_mm256_add_pd(_mm256_sub_pd(vp02,_mm256_sub_pd(vah11,val02)),_mm256_sub_pd(vp11,val02));
    // [p02, ah11] = Calc.twoSum(ah11, p20);
    vp02=_mm256_add_pd(vah11,vp20);
    val02=_mm256_sub_pd(vp02,vah11);
    vah11= _mm256_add_pd(_mm256_sub_pd(vah11,_mm256_sub_pd(vp02,val02)),_mm256_sub_pd(vp20,val02));
    // [p11, p20] = Calc.twoSum(bh02, ah11);
    vp11=_mm256_add_pd(vbh02,vbh02);
    val02=_mm256_sub_pd(vp11,vbh02);
    vp20=_mm256_add_pd(_mm256_sub_pd(vbh02,_mm256_sub_pd(vp11,val02)),_mm256_sub_pd(vah11,val02));
    
    // two_sum(&c2[0], &p10, p02, p10);
    vzlh = _mm256_add_pd(vp02 , vp10);
    vbl02 = _mm256_sub_pd(vzlh , vp02);
    vp10 = _mm256_add_pd(_mm256_sub_pd(vp02 , _mm256_sub_pd(vzlh , vbl02)) , _mm256_sub_pd(vp10 , vbl02));
    //two_sum(&p11, &bl00, p11, bl00);
    vbl20=vp11;
    vp11 = _mm256_add_pd(vbl20 , vbl00);
    vbl01 = _mm256_sub_pd(vp11 , vbl20);
    vbl00 = _mm256_add_pd(_mm256_sub_pd(vbl20 , _mm256_sub_pd(vp11 , vbl01)) , _mm256_sub_pd(vbl00 , vbl01));
    //two_sum(&p10, &p11, p10, p11);
    vbl01=vp10;
    vp10 = _mm256_add_pd(vbl01 , vp11);
    vbl20 = _mm256_sub_pd(vp10 , vbl01);
    vp11 = _mm256_add_pd(_mm256_sub_pd(vbl01 , _mm256_sub_pd(vp10 , vbl20)) , _mm256_sub_pd(vp11 , vbl20));
    //O(eps^4) terms
    val10 = _mm256_add_pd(_mm256_add_pd(val10 , vp20) , _mm256_add_pd(vbl00 , vp11));
    
    //no fma
    //O(eps^3) terms
    vzll =_mm256_add_pd(
            _mm256_add_pd(
            _mm256_add_pd(vp10,
            _mm256_add_pd(
            _mm256_add_pd(
            _mm256_add_pd(
            _mm256_mul_pd(vxhh,vyll),
            _mm256_mul_pd(vxhl,vylh)),
            _mm256_mul_pd(vxlh,vyhl)),
            _mm256_mul_pd(vxll,vyhh))),
            _mm256_add_pd(vah02,vbl11)),val11);
    
    val00=vzhh;
    vbl00=vzhl;
    vah02=vzlh;
    val11=vzll;
    
    //renormalize(&c0[0], &c1[0], &c2[0], &c3[0], al00,bl00,ah02,al11, al10);
    vbl02 = _mm256_add_pd(val11 , val10);
    vbl01 = _mm256_sub_pd(val10 , _mm256_sub_pd(vbl02, val11));
    vbh02 = _mm256_add_pd(vah02 , vbl02);
    vbh01 = _mm256_sub_pd(vbl02 , _mm256_sub_pd(vbh02 , vah02));
    vbl02  = _mm256_add_pd(vbl00 , vbh02);
    val01 = _mm256_sub_pd(vbh02, _mm256_sub_pd(vbl02 , vbl00));
    vzhh = _mm256_add_pd(val00 , vbl02);
    vah01 = _mm256_sub_pd(vbl02 , _mm256_sub_pd(vzhh , val00));
    vbl02 = _mm256_add_pd(vbh01 , vbl01);
    vbh01 = _mm256_sub_pd(vbl01 ,_mm256_sub_pd(vbl02 , vbh01));
    vbh02 = _mm256_add_pd(val01 , vbl02);
    val01 = _mm256_sub_pd(vbl02 , _mm256_sub_pd(vbh02 ,val01));
    vzhl = _mm256_add_pd(vah01 , vbh02);
    vah01 = _mm256_sub_pd(vbh02 , _mm256_sub_pd(vzhl, vah01));
    vbl02 = _mm256_add_pd(val01 , vbh01);
    val01 = _mm256_sub_pd(vbh01 , _mm256_sub_pd(vbl02 ,val01));
    vzlh = _mm256_add_pd(vah01 , vbl02);
    vah01 = _mm256_sub_pd(vbl02 , _mm256_sub_pd(vzlh , vah01));
    vzll = _mm256_add_pd(vah01 , val01);
    
    *vchh = vzhh;
    *vchl = vzhl;
    *vclh = vzlh;
    *vcll = vzll;
}
