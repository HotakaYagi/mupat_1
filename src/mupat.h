
/**
 * @file mupat.h
 * @brief hedder file of MuPAT
 * @author Hotaka Yagi
 * @date last update 2020 Jan 15
 */


/*! \mainpage MuPAT an open-source interactive multiple precision arithmetic toolbox
 * 
 * \section log_sec Change log
 * 2019 Apr 10 MuPAT 3 released
 * 2020 Jan 15 MuPAT 3 updated to 3.1
 *
 * \section intro_sec Introduction
 *
 * This project helps user to compute fastly some matrix and vector operation with high-precision.\n
 * While rounding errors are unavoidable in floating point arithmetic, they can be reduced through the use of high-precision arithmetic. \n
 * Easy-to-use high-precision arithmetic is also important for end users. \n
 * In response to this need, our team developed MuPAT, an open-source interactive multiple precision arithmetic toolbox, which runs independently on any hardware and operating system for use with the MATLAB and Scilab computing environments[1].\n
 * Currently, you can get Scilab version at https://atoms.scilab.org/toolboxes/DD_QD \n
 * \n
 * MuPAT uses double-double and quad-double[2] algorithms defined by a double precision arithmetic combination. 
 *
 * - \subpage DDQD "details of double-double and quad-double arithmetic"
 *
 * However, since these algorithms require from 10 to 600 double precision floating point operations for each operation, excessively long computation times often result. \n
 * Therefore, in order to accelerate processing, we selected three parallel processing features, FMA[3], AVX2[3], and OpenMP[4] and then implemented these three features in MuPAT in order to examine a number of basic matrix and vector operations. 
 *
 * - \subpage FMA "What is FMA?"
 *
 * - \subpage AVX2 "What is AVX2?"
 *
 * - \subpage OpenMP "What is OpenMP?"
 *
 * We then evaluated the performance of MuPAT on a personal computer using a roofline model[5] and found that it allowed those matrix and vector operations to reach nearly 60% of their theoretically attainable performance levels.
 * 
 * \section use_sec How to use it?
 * You can use MuPAT if you have MATLAB.\n
 * Documentation is attached with.
 *
 * \section trob_sec Trouble shooting
 * Please e-mail for 1419521@ed.tus.ac.jp
 *
 * 
 *
 * \section info_sec Developer information
 *
 * Author: Hotaka Yagi \n
 * Contact: 1419521@ed.tus.ac.jp \n 
 * Affiliation: Master's course at Tokyo univ. of science
 */ 

//-----------------------------------------------------------

/*! \page DDQD details of double-double and quad-double arithmetic
 * Double-double number \f$\alpha\f$ is represented by 2 double precision numbers \f$\alpha_0 \f$ and \f$ \alpha_1 \f$ as below.\n
 * \f$ \alpha = \alpha_0 + \alpha_1 \f$ \n
 * \f$ \alpha_0 \f$ is the higher part of \f$ \alpha \f$ , and
 * \f$ \alpha_1 \f$ is the lower part of \f$ \alpha \f$ \n
 * 
 * Here, \f$ \alpha_0 \f$ and \f$ \alpha_1 \f$ satisfy \f$ |\alpha_1| \leq  \frac{1}{2}ulp(\alpha_0) \f$ \n
 * ulp means units in the last place.
 *
 * Likewisely, quad-double number \f$\beta\f$ is represented by 4 double precision numbers \f$\beta_0 \f$ , \f$ \beta_1 \f$ , \f$\beta_2 \f$ , and  \f$ \beta_3 \f$  as below.\n
 * \f$ \beta = \beta_0 + \beta_1 + \beta_2 + \beta_3 \f$ \n
 * Here, \f$ \beta_0 \f$ to \f$ \beta_3 \f$ satisfy \f$ |\beta_{i+1}| \leq  \frac{1}{2}ulp(\alpha_{i}), (i=0,1,2) \f$ \n
 */

//-----------------------------------------------------------

/*! \page FMA What is FMA?
 * FMA refers to fused multiply add.\n
 * A floating point multiply-add operation is performed in one step via a single rounding FMA instruction. \n
 * We applied twoprod_fma[2] algorithm that computes the exact product of two floating point numbers.
 *
 */

//-----------------------------------------------------------

/*! \page AVX2 What is AVX2?
 * AVX2 refers to Advanced Vector EXtensions2.\n
 * Since four double precision data are processed simultaneously with the AVX2 instruction, we can expect the computation time to become four times faster.\n
 * When the array length is not a multiple of SIMD register length, the leftovers must be handled without AVX2.\n
 * When we compute the inner product, we sum up the four SIMD register elements after the main loop. 
 */

//-----------------------------------------------------------

/*! \page OpenMP What is OpenMP?
 *
 * Since OpenMP enables thread level parallelism within a shared memory, we can expect the computation time to decrease by a multiple of the number of cores.\n
 * In our codes, the OpenMP 'pragma omp for' directives are used in the parallel region and the number of threads is fixed to the number of the cores.
 */

//-----------------------------------------------------------

/*! \page page1 References
 *
 * [1] S. Kikkawa, T. Saito, E. Ishiwata, and H. Hasegawa, Development and acceleration of multiple precision arithmetic toolbox MuPAT for Scilab, JSIAM Letters, Vol.5, pp.9-12 (2013).\n
 * [2] Y. Hida, X. S. Li, and D. H. Baily, Quad-double arithmetic: algorithms, implementation and application, Technical Report LBNL-46996, Lawrence Berkeley National Laboratory(2000).\n
 * [3] Intel Intrinsics Guide, https://software.intel.com/sites/landingpage/IntrinsicsGuide/ \n
 * [4] OpenMP Home, https://www.openmp.org/ \n
 * [5] S. Williams, A. Waterman, and D. Patterson, Roofline: An insightful visual performance model for multicore architectures, Communications of the ACM, Vol.52(4), pp.65-76 (2009).
 */

/*! \page page2 Co workers
 * 
 * Emiko ISHIWATA
 * (Tokyo univ. of science)\n
 * https://www.tus.ac.jp/en/fac/p/index.php?1587  \n
 *
 * Hidehiko HASEGAWA
 * (Tsukuba univ.) \n
 * http://www.slis.tsukuba.ac.jp/~hasegawa.hidehiko.ga/ \n
 *
 */

/*! \page page3 ToDo list
 * 
 * Implemetation of sparse type.\n
 *
 */

//-----------------------------------------------------------

#ifndef __INCLUDE_MUPAT_H__
#define __INCLUDE_MUPAT_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <x86intrin.h>

//-----------------------------------------------------------------
// Definition of vector and matrix operation
//-----------------------------------------------------------------

//-----------------------------------------------------------------
// Double - Double-Double operation 
//-----------------------------------------------------------------
void _d_a_dd(double*,double*,double*,double*,double*,int,int,int,int);
void _d_scl_dd(double*,double*,double*,double*,double*,int,int,int,int,int);
void _d_dot_dd(double*,double*,double*,double*,double*,int,int,int,int,int);
void _d_mv_dd(double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _d_tmv_dd(double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _d_mm_dd(double*,double*,double*,double*,double*,int,int,int,int,int,int);

//-----------------------------------------------------------------
// Double - Quad-Double operation 
//-----------------------------------------------------------------
void _d_a_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int);
void _d_scl_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _d_dot_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _d_mv_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _d_tmv_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _d_mm_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);

//-----------------------------------------------------------------
// Double-Double - Double operation 
//-----------------------------------------------------------------
void _dd_scl_d(double*,double*,double*,double*,double*,int,int,int,int,int);
void _dd_mv_d(double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _dd_tmv_d(double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _dd_mm_d(double*,double*,double*,double*,double*,int,int,int,int,int,int);

//-----------------------------------------------------------------
// Double-Double - Double-Double operation 
//-----------------------------------------------------------------
void _dd_a_dd(double*,double*,double*,double*,double*,double*,int,int,int,int);
void _dd_a_dd_ieee(double*,double*,double*,double*,double*,double*,int,int,int,int);
void _dd_scl_dd(double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _dd_dot_dd(double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _dd_dot_dd_ieee(double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _dd_mv_dd(double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _dd_mv_dd_ieee(double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _dd_tmv_dd(double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _dd_mm_dd(double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _dd_mm_dd_ieee(double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);

//-----------------------------------------------------------------
// Double-Double - Quad-Double operation 
//-----------------------------------------------------------------
void _dd_a_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int);
void _dd_scl_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _dd_dot_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _dd_mv_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _dd_tmv_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _dd_mm_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);

//-----------------------------------------------------------------
// Quad-Double - Double operation 
//-----------------------------------------------------------------
void _qd_scl_d(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _qd_mv_d(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _qd_tmv_d(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _qd_mm_d(double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);

//-----------------------------------------------------------------
// Quad-Double - Double-Double operation 
//-----------------------------------------------------------------
void _qd_scl_dd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _qd_mv_dd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _qd_tmv_dd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _qd_mm_dd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);

//-----------------------------------------------------------------
// Quad-Double - Quad-Double operation 
//-----------------------------------------------------------------
void _qd_a_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int);
void _qd_scl_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _qd_axpy_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _qd_dot_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _qd_dot_qd_self(double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int);
void _qd_mv_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _qd_tmv_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);
void _qd_mm_qd(double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int,int,int,int,int,int);

//-----------------------------------------------------------------
// Definition of base operation
//-----------------------------------------------------------------

//-----------------------------------------------------------------
// Serial
//-----------------------------------------------------------------
void _d_add_dd(double*,double*,double,double,double);
void _d_mul_dd(double*,double*,double,double,double);
void _d_add_qd(double*,double*,double*,double*,double,double,double,double,double);
void _d_mul_qd(double*,double*,double*,double*,double,double,double,double,double);
void _dd_add_dd(double*,double*,double,double,double,double);
void _dd_add_dd_ieee(double*,double*,double,double,double,double);
void _dd_mul_dd(double*,double*,double,double,double,double);
void _dd_add_qd(double*,double*,double*,double*,double,double,double,double,double,double);
void _dd_mul_qd(double*,double*,double*,double*,double,double,double,double,double,double);
void _qd_add_qd(double*,double*,double*,double*,double,double,double,double,double,double,double,double);
void _qd_mul_qd(double*,double*,double*,double*,double,double,double,double,double,double,double,double);

//-----------------------------------------------------------------
// FMA
//-----------------------------------------------------------------
void _d_mul_dd_fma(__m256d*,__m256d*,__m256d,__m256d,__m256d);
void _d_mul_qd_fma(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d);
void _dd_mul_dd_fma(__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d);
void _dd_mul_qd_fma(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d);
void _qd_mul_qd_fma(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d);

//-----------------------------------------------------------------
// AVX2
//-----------------------------------------------------------------
void _d_add_dd_avx2(__m256d*,__m256d*,__m256d,__m256d,__m256d);
void _d_mul_dd_avx2(__m256d*,__m256d*,__m256d,__m256d,__m256d);
void _d_add_qd_avx2(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d);
void _d_mul_qd_avx2(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d);
void _dd_add_dd_avx2(__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d);
void _dd_add_dd_avx2_ieee(__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d);
void _dd_mul_dd_avx2(__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d);
void _dd_add_qd_avx2(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d);
void _dd_mul_qd_avx2(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d);
void _qd_add_qd_avx2(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d);
void _qd_mul_qd_avx2(__m256d*,__m256d*,__m256d*,__m256d*,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d,__m256d);
#endif 