function obj=ddeye(m,n)
if nargin==0
    obj=DD(eye(),zeros());
elseif nargin == 1
    obj=DD(eye(m),zeros(m));
else
    obj=DD(eye(m,n),zeros(m,n));
end
end