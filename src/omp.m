function omp(n)
global defthreadNum;
if n < 1
    defthreadNum = 1;
else
    defthreadNum = n;
end
name = 'set the number of threads = ';
X = [name,num2str(n)];
disp(X);
end