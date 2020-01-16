function obj=qdeye(m,n)
if nargin==0
    obj=QD(eye(),zeros(),zeros(),zeros());
elseif nargin == 1
    obj=QD(eye(m),zeros(m),zeros(m),zeros(m));
else
    obj=QD(eye(m,n),zeros(m,n),zeros(m,n),zeros(m,n));
end
end