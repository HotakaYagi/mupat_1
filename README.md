# MuPAT
2019 Apr 10 MuPAT 3 released  
2020 Jan 15 updated to MuPAT 3.1
## Introduction  
This project helps user to compute fastly some matrix and vector operation with high-precision.  
While rounding errors are unavoidable in floating point arithmetic, they can be reduced through the use of high-precision arithmetic.  
Easy-to-use high-precision arithmetic is also important for end users. In response to this need, our team developed MuPAT, an open-source interactive multiple precision arithmetic toolbox, which runs independently on any hardware and operating system for use with the MATLAB and Scilab computing environments[1].  
Currently, you can get Scilab version at https://atoms.scilab.org/toolboxes/DD_QD   

MuPAT uses double-double and quad-double[2] algorithms defined by a double precision arithmetic combination.  

However, since these algorithms require from 10 to 600 double precision floating point operations for each operation, excessively long computation times often result.   

Therefore, in order to accelerate processing, we selected three parallel processing features, FMA[3], AVX2[3], and OpenMP[4] and then implemented these three features in MuPAT in order to examine a number of basic matrix and vector operations.  


We then evaluated the performance of MuPAT on a personal computer using a roofline model[5] and found that it allowed those matrix and vector operations to reach nearly 60% of their theoretically attainable performance levels.  

## How to use it?
You can use MuPAT if you have MATLAB.  
Documentation is attached with.

## Reference Manual
You can obtain detail via doxygen at maincode directory.  
http://www.doxygen.nl/

## References
 [1] S. Kikkawa, T. Saito, E. Ishiwata, and H. Hasegawa, Development and acceleration of multiple precision arithmetic toolbox MuPAT for Scilab, JSIAM Letters, Vol.5, pp.9-12 (2013).  
 [2] Y. Hida, X. S. Li, and D. H. Baily, Quad-double arithmetic: algorithms, implementation and application, Technical Report LBNL-46996, Lawrence Berkeley National Laboratory(2000).  
 [3] Intel Intrinsics Guide, https://software.intel.com/sites/landingpage/IntrinsicsGuide/  
 [4] OpenMP Home, https://www.openmp.org/  
 [5] S. Williams, A. Waterman, and D. Patterson, Roofline: An insightful visual performance model for multicore architectures, Communications of the ACM, Vol.52(4), pp.65-76 (2009).  


## Trouble Shooting
 Please e-mail for 1419521@ed.tus.ac.jp

## Developer Information
Author: Hotaka Yagi   
Contact: 1419521@ed.tus.ac.jp   
Affiliation: Master's course at Tokyo univ. of science  