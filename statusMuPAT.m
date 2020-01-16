function statusMuPAT
global deffma;
global defavx;
global defthreadNum;

fma = deffma;
avx = defavx;
numth = defthreadNum;
fma_now = 'off';
avx_now = 'off';
omp_now = '1thread (serial)';

if fma == 1
    fma_now = 'on';
end

if avx == 1
    avx_now = 'on';
end

if numth ~= 1
    omp_now = num2str(numth);
end    

x = ['Cureently MuPAT is running at the number of threads: ', omp_now, ', AVX is ', avx_now, ', and FMA is ', fma_now];

disp(x);
end