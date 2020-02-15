function startMuPAT(threadNum, avx, fma)
global deffma;
global defavx;
global defthreadNum;
global defacc;
defacc = 0;
addpath src
switch nargin
    case 0
        deffma = 0;
        defavx = 0;
        defthreadNum = 1;
    case 1
        deffma = 0;
        defavx = 0;
        if threadNum < 1
            defthreadNum = 1;
        else
            defthreadNum = threadNum;
        end
    case 2
        deffma = 0;
        defavx = avx;
        if threadNum < 1
            defthreadNum = 1;
        else
            defthreadNum = threadNum;
        end
    case 3
        deffma = fma;
        defavx = avx;
        if threadNum < 1
            defthreadNum = 1;
        else
            defthreadNum = threadNum;
        end
    otherwise
        disp("expect 0 ~ 3 input");
end
end