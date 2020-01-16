function b = qdinput(a)
digits = '0123456789';
id = 0;
ip = -1;
is = 0;
inz = 0;
s = QD();

for i = 1 : length(a)
    ci = a(i);
    if (ci ==' ' && id == 0)
    elseif ci == '.'
        if(ip >= 0)
            abort;
        end
        ip = id;
        inz = 1;
    elseif ci =='+'
        if(id ~= 0 || ip >= 0 || is ~= 0)
            abort;
        end
        is = 1;
    elseif ci =='-'
        if(id ~= 0 || ip >= 0 || is ~= 0)
            abort;
        end
        is = -1;
    elseif (ci == 'e' || ci == 'E' || ci =='d' || ci =='D')
        break;
    elseif(contains(digits, ci) == 0)%check
        abort;
    else
        bi = strfind(digits, ci) - 1;
        if((inz > 0) || (bi > 0))
            inz = 1;
            id = id + 1;
            s = s * 10;
            s  = s + QD(bi);
        end
    end
end
if(is == -1)
    s.hi = -s.hi;
    s.lo = -s.lo;
end
k = i;
if(ip == -1)
    ip = id;
end

ie = 1;
is = 0;
ca = '';

for i = k+1 : length(a)
    ci = a(i);
    if (ci == ' ')
    elseif(ci == '+')
        if(ie ~= 0 || is ~= 0)
            abort;
        end
        is = 1;
    elseif(ci == '-')
        if(ie ~= 0 || is ~= 0)
            abort;
        end
        is = -1;
    elseif(contains(digits, ci) == 0)
        abort;
    else
        ie = ie + 1;
        if (ie >= 3)
            abort;
        end
        ca(ie) = ci;
    end
end

ie = 0;
for i = 1:length(ca)
    k = strfind(digits, ci) - 1;
    if(k<0)
        abort;
    elseif(k<=9)
        ie = 10 * ie + k;
    end
end

if (is == -1)
    ie = -ie;
end
ie = ie + ip - id;
s2 = QD(10);
s2 = s2^ie;
b = s * s2;
end