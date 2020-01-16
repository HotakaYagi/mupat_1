function obj=qdrand(m,n)
%if you want to use this function, you should open @Calc file
if nargin==0
    a1 = rand();
    a2 = rand()*1d-15;
    a3 = rand()*1d-30;
    a4 = rand()*1d-45;
    a5 = rand()*1d-60;
    [a,b,c,d] = renormalize(a1,a2,a3,a4,a5);
    obj=QD(a,b,c,d);
elseif nargin == 1
    a1 = rand(m);
    a2 = rand(m)*1d-15;
    a3 = rand(m)*1d-30;
    a4 = rand(m)*1d-45;
    a5 = rand(m)*1d-60;
    [a,b,c,d] = renormalize(a1,a2,a3,a4,a5);
    obj=QD(a,b,c,d);
elseif nargin == 2
    a1 = rand(m,n);
    a2 = rand(m,n)*1d-15;
    a3 = rand(m,n)*1d-30;
    a4 = rand(m,n)*1d-45;
    a5 = rand(m,n)*1d-60;
    [a,b,c,d] = renormalize(a1,a2,a3,a4,a5);
    obj=QD(a,b,c,d);
else
end
end

function [b1,b2,b3,b4] = renormalize(a1,a2,a3,a4,a5)
s = a4 + a5;
t4 = a5 - (s - a4);
ss = a3 + s;
t3 = s - (ss - a3);
s  = a2 + ss;
t2 = ss - (s - a2);
b1 = a1 + s;
t1 = s - (b1 - a1);
s = t3 + t4;
t3 = t4 - (s - t3);
ss = t2 + s;
t2 = s - (ss - t2);
b2 = t1 + ss;
t1 = ss - (b2 - t1);
s = t2 + t3;
t2 = t3 - (s -t2);
b3 = t1 + s;
t1 = s - (b3 - t1);
b4 = t1 + t2;
end