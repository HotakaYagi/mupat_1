function obj=ddrand(m,n)
if nargin==0
    a = rand();
    b = rand()*10^-10;
    s = a + b;
    b = b - (s - a);
    obj=DD(s,b);
elseif nargin == 1
    a = rand(m);
    b = rand(m)*10^-10;
    s = a + b;
    b = b - (s - a);
    obj=DD(s,b);
else
    a = rand(m,n);
    b = rand(m,n)*10^-10;
    s = a + b;
    b = b - (s - a);
    obj=DD(s,b);
end
end