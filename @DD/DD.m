classdef  DD < matlab.mixin.CustomDisplay
    properties(GetAccess = public, SetAccess = private,Hidden = false)
        hi = 0;
        lo = 0;
    end
    methods (Static)
        function out = ddformat(varargin)
            persistent style
            if(isempty(style))
                %preset is 0
                style = 0;
            end
            
            if(isempty(varargin))
                %no arguments mean return current value
                out = style;
            elseif(length(varargin) == 1)
                %set arguments in style
                style = varargin{1};
            end
        end
    end
    methods (Access = protected)
        %change disply of properties
        function propgrp = getPropertyGroups(obj)
            switch(DD.ddformat())
                case 0
                    propList = struct('hi',obj.hi,'lo',obj.lo);
                    propgrp = matlab.mixin.util.PropertyGroup(propList);
                    %regular output
                case 1
                    %in case scalar, use ddprint
                    [m, n] = size(obj.hi);
                    if(m == 1 && n == 1)
                        %for scalar
                        propList = struct('DD',ddprint(obj));
                        propgrp = matlab.mixin.util.PropertyGroup(propList);
                    elseif(m == 1 || n == 1)
                        %for vector
                        propList = struct('DD',strcat('[',num2str(m),'×', num2str(n), '  vector ]') );
                        propgrp = matlab.mixin.util.PropertyGroup(propList);
                    else
                        %for matrix
                        propList = struct('DD',strcat('[',num2str(m),'×', num2str(n), '  matrix ]') );
                        propgrp = matlab.mixin.util.PropertyGroup(propList);
                    end
            end
        end
        %change header in disply of properties
        function header = getHeader(obj)
            header = sprintf('');
        end
    end
    methods
        function obj = DD(a,b)
            if(nargin == 1)
                [row_a,col_a] = size(a);
                if isa(a,'DD')
                    obj.hi=a.hi;
                    obj.lo=a.lo;
                elseif isa(a,'QD')
                    obj.hi=a.hh;
                    obj.lo=a.hl;
                else
                    obj.hi = a;
                    obj.lo = zeros(row_a, col_a);
                end
            elseif(nargin==2)
                obj.hi = a;
                obj.lo = b;
            elseif(nargin==0)
                obj.hi=0;
                obj.lo=0;
            end
        end
        function obj = dd(a,b)
            obj=DD(a,b);
        end
        %----------------------%
        %      base method     %
        %----------------------%
        %size
        function [m, n] = size(obj)
            [m, n] = size(obj.hi);
        end
        %double
        function d = double(obj)
            d = obj.hi;
        end
        
        function d = Double(obj)
            d = obj.hi;
        end
        
        function obj=sparse(A)
            obj=DD(sparse(A.hi),sparse(A.lo));
        end
        
        function obj=full(A)
            obj=DD(full(A.hi),full(A.lo));
        end
        
        %sign
        function X=sign(A)
            [m,n]=size(A);
            X = zeros(m, n);
            for i = 1:m
                for j = 1:n
                    if A(i, j) < 0
                        X(i, j) = -1;
                    elseif A(i, j) > 0
                        X(i, j) = 1;
                    end
                end
            end
        end
        
        %mod
        function a=mod(a,b)
            if b<0
                b=-b;
            end
            if a>0
                while a-b>0
                    a=a-b;
                end
            else
                while a<0
                    a=a+b;
                end
                a=a-b;
            end
        end
        
        %abs
        function obj = abs(a)
            obj=a;
            c = find(sign(a.hi) == -1);
            obj.hi(c) = (-1) * a.hi(c);
            obj.lo(c) = (-1) * a.lo(c);
        end
        
        function obj = sqrt(a)
            [m,n]=size(a.hi);
            obj = DD(zeros(m,n));
            warning('off','MATLAB:structOnObject');
            a=struct(a);
            for i=1:m
                for j=1:n
                    if a.hi(i,j)==0&&a.lo(i,j)==0
                        obj.hi(i,j)=0;
                        obj.lo(i,j)=0;
                        continue
                    end
                    c = sqrt(a.hi(i,j));
                    [p, e] = Calc.twoProd(c, c);
                    cc =(a.hi(i,j) - p - e + a.lo(i,j)) * 0.5 / c;
                    [obj.hi(i,j),obj.lo(i,j)] = Calc.fastTwoSum(c, cc);
                end
            end
        end
        
        function z = dot(x, y)
            global deffma;
            global defavx;
            global defthreadNum;
            global defacc;
            acc = defacc;
            fma = deffma;
            avx = defavx;
            threadNum = defthreadNum;
            isxd = isnumeric(x);
            isyd = isnumeric(y);
            isxdd = isa(x, 'DD');
            isydd = isa(y, 'DD');
            if(isxd)
                if(isydd)
                    [z.hi,z.lo]=d_dot_dd(x,y.hi,y.lo,threadNum,avx,fma);
                    z=DD(z.hi,z.lo);
                end
            elseif(isxdd)
                if(isyd)
                    [z.hi,z.lo]=d_dot_dd(y,x.hi,x.lo,threadNum,avx,fma);
                    z=DD(z.hi,z.lo);
                elseif(isydd)
                    if acc == 0
                        [z.hi,z.lo]=dd_dot_dd(x.hi,x.lo,y.hi,y.lo,threadNum,avx,fma);
                        z=DD(z.hi,z.lo);
                    elseif acc == 1
                        [z.hi,z.lo]=dd_dot_dd_ieee(x.hi,x.lo,y.hi,y.lo,threadNum,avx,fma);
                        z=DD(z.hi,z.lo);
                    end
                end
            end
        end
        
        function z = tmv(x, y)
            global deffma;
            global defavx;
            global defthreadNum;
            global defacc;
            acc = defacc;
            fma = deffma;
            avx = defavx;
            threadNum = defthreadNum;
            isxd = isnumeric(x);
            isyd = isnumeric(y);
            isxdd = isa(x, 'DD');
            isydd = isa(y, 'DD');
            if(isxd)
                if(isydd)
                    [z.hi,z.lo]=d_tmv_dd(x,y.hi,y.lo,threadNum,avx,fma);
                    z=DD(z.hi,z.lo);
                end
            elseif(isxdd)
                if(isyd)
                    [z.hi,z.lo]=d_tmv_dd(x.hi,x.lo,y,threadNum,avx,fma);
                    z=DD(z.hi,z.lo);
                elseif(isydd)
                    if acc == 0
                        [z.hi,z.lo]=dd_tmv_dd(x.hi,x.lo,y.hi,y.lo,threadNum,avx,fma);
                        z=DD(z.hi,z.lo);
                    elseif acc == 1
                        [z.hi,z.lo]=dd_tmv_dd_ieee(x.hi,x.lo,y.hi,y.lo,threadNum,avx,fma);
                        z=DD(z.hi,z.lo);
                    end
                end
            end
        end
        
        
        %norm
        function obj=norm(x,N)
            [n,m]=size(x);
            if m~=1
                disp('set vector');
            end
            switch(N)
                case 1
                    obj=DD(0);
                    x=abs(x);
                    x=struct(x);
                    for i=1:n
                        obj=obj+DD(x.hi(i),x.lo(i));
                    end
                case 2
                    obj = sqrt(dot(x,x));
                case 'inf'
                    x=abs(x);
                    x=struct(x);
                    x_max=DD(x.hi(1),x.lo(1));
                    for i=1:n
                        if(DD(x.hi(i),x.lo(i)) > x_max)
                            x_max = DD(x.hi(i),x.lo(i));
                        end
                    end
                    obj = x_max;
                case 'fro'
                    obj=DD(0);
                    x=struct(x);
                    for i=1:n
                        for j=1:m
                            obj = obj + DD(x.hi(i,j),x.lo(i,j))*DD(x.hi(i,j),x.lo(i,j));
                        end
                    end
                    obj=sqrt(obj);
                otherwise
                    disp('The only matrix norms available are 1, 2, inf, and fro');
            end
        end
        %a^n
        function obj=mpower(a,n)
            if mod(n,1)~=0
                disp('a^n (n is integer)')
            else
                if n==0
                    obj=DD(0);
                elseif n>0
                    obj=ddpow(a,n);
                else
                    obj=ddpow(1/a,-n);
                end
            end
        end
        %ddpow
        function b = ddpow(a,n)
            b = DD(0);
            absn = abs(n);
            if(a.hi == 0)
                if(n > 0)
                    b = DD(0);
                else
                    error(27) %Divide by zero...
                end
            else
                if(absn == 0)
                    b = DD(1);
                elseif(absn == 1)
                    b = a;
                elseif(absn == 2)
                    b = a*a;
                else
                    % Determine the least integer MN such that 2^MN > MN
                    mn = fix(log2(exp(1))*log(absn) + 1.0 + 1.0D-14);
                    s0 = a;
                    b = DD(1);
                    kn = absn;
                    % Compute B^N using binary rule for exponentiation
                    for j=1:mn
                        kk = fix(kn / 2);
                        if(kn ~= 2.0*kk)
                            s1 = b * s0;
                            b = s1;
                        end
                        kn = kk;
                        if(j < mn)
                            s1 = s0 * s0;
                            s0 = s1;
                        end
                    end
                end
                % Compute reciprocal if N is negative
                if(n < 0)
                    b = 1.0 / b;
                end
            end
        end
        
        function b=length(a)
            b=size(a);
        end
        
        function obj=triu(a,k)
            obj=DD(triu(a.hi,k),triu(a.lo,k));
        end
        
        function obj=diag(a)
            obj=DD(diag(a.hi),diag(a.lo));
        end
        
        %ceil
        function b=ceil(a)
            [m,n] = size(a.hi);
            s = ceil(a.hi);
            t = zeros(m,n);
            b = DD(s,t);
            if(s == a.hi)
                t = ceil(a.lo);
                b.hi = s + t;
                b.lo = t - (b.hi - s);
            end
        end
        
        %floor
        function b=floor(a)
            [m,n] = size(a.hi);
            s = floor(a.hi);
            t = zeros(m,n);
            b = DD(s,t);
            if(s == a.hi)
                t = floor(a.lo);
                b.hi = s + t;
                b.lo = t - (b.hi - s);
            end
        end
        
        %ddprint
        function ddprint(A)
            ln = 32;
            ib = zeros(ln,1);
            digits = ['0';'1';'2';'3';'4';'5';'6';'7';'8';'9'];
            f = DD(10);
            s = DD(0);
            [m, n] = size(A.hi);
            C = cell(m,n);
            a = DD(0);
            for x = 1:m
                for y = 1:n
                    a.hi = A.hi(x,y);
                    a.lo = A.lo(x,y);
                    % Determine exact power of ten for exponent.
                    if(a.hi ~= 0)
                        t1 = log10(abs(a.hi));
                        if(t1 >= 0)
                            nx = fix(t1);
                        else
                            nx = fix(t1) - 1;
                        end
                        s = a /(ddpow(f,nx));
                        
                        if(s.hi < 0)
                            s.hi = -s.hi;
                            s.lo = -s.lo;
                        end
                        
                        % If we didn't quite get it exactly right,multiply or divide by 10 to fix.
                        i = 0;
                        r = (1 <= s.hi) &&( s.hi < 10);
                        
                        if((1 <= s.hi) &&( s.hi < 10))
                            bool = false;
                        else
                            bool = true;
                        end
                        
                        while((s.hi < 1) || (s.hi >= 10 )|| (bool == true))
                            i = i + 1;
                            if(s.hi < 1)
                                nx = nx - 1;
                                s = s * 10;
                                if(i > 3)
                                    bool = false;
                                end
                            elseif(s.hi >= 10)
                                nx = nx + 1;
                                s = s / 10;
                            end
                        end
                    else
                        nx = 0;
                    end
                    % compute digits
                    for i=1:ln
                        ib(i) = int16(fix(s.hi));
                        s = (s - double(ib(i))) * 10;
                    end
                    % fix negative digits
                    
                    for i=ln:-1:2
                        if(ib(i) < 0)
                            ib(i) = ib(i) + 10;
                            ib(i-1) = ib(i-1) - 1;
                        end
                    end
                    
                    if(ib(1) < 0)
                        disp('ddprint : negative leading digit')
                        c = 0;
                    end
                    
                    % round
                    
                    %                     if(ib(ln) >= 5) then
                    %                         ib(ln-1) = ib(ln-1) + 1;
                    %
                    %                         for i=ln-1:-1:2
                    %                             if(ib(i) == 10) then
                    %                                 ib(i) = 0;
                    %                                 ib(i-1) = ib(i-1) + 1;
                    %                             end
                    %                         end
                    %
                    %                         if(ib(1) == 10) then
                    %                             ib(1) = 1;
                    %                             nx = nx + 1;
                    %                         end
                    %                     end
                    
                    %  insert digit characters in ib
                    
                    c = '';
                    if(a.hi >= 0)
                        c = [c,''];
                    else
                        c = [c,'-'];
                    end
                    
                    c = [c,digits(ib(1)+1)];
                    c = [c,'.'];
                    for i=2:ln-1
                        c = [c,digits(ib(i)+1)];
                    end
                    
                    % insert exponent
                    
                    c =[c,'E'];
                    is = sign(nx);
                    d1 = abs(nx);
                    
                    if(is < 0)
                        c = [c,'-'];
                    end
                    ca = '';
                    for i=1:4
                        d2 = int16(fix(d1/10));
                        k = 1 + (d1 - 10*d2);
                        d1 = d2;
                        ca(i,1) = digits(k);
                        if(d1 == 0)
                            break;
                        end
                    end
                    for j=i:-1:1
                        c = [c,ca(j,1)];
                    end
                    C(x, y) = cellstr(c);
                end
            end
            disp(C);
        end
        
        function obj=ddeye(m,n)
            if nargin==0
                obj=DD(eye(),zeros());
            elseif nargin == 1
                obj=DD(eye(m),zeros(m));
            else
                obj=DD(eye(m,n),zeros(m,n));
            end
        end
        
        function obj=ddpi()
            obj= DD(3.141592653589793116e+00,1.224646799147353207e-16);
        end
        
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
        
        %----------------------%
        %       operator       %
        %----------------------%
        %plus DD <- A + B
        function obj = plus(a, b)
            global defavx;
            global defthreadNum;
            global defacc;
            acc = defacc;
            avx = defavx;
            threadNum = defthreadNum;
            obj = DD(0);
            
            [m1,n1]=size(a);
            [m2,n2]=size(b);
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            if(m1==m2&&n1==n2)
                if(isD1)
                    if(isDD2) %D+DD
                        [obj.hi, obj.lo] = d_a_dd(a,b.hi,b.lo,threadNum,avx);
                    else
                        %obj2 is not supported
                    end
                else
                    if(isDD1)
                        if(isD2) %DD+D
                            [obj.hi, obj.lo] = d_a_dd(b,a.hi,a.lo,threadNum,avx);
                            
                            %DD+DD
                        elseif(isDD2)
                            if acc == 0
                                if(m1==1&&n1==1)
                                    [sh, eh] = Calc.twoSum(a.hi, b.hi);
                                    [sl, el] = Calc.twoSum(a.lo, b.lo);
                                    [sh, eh] = Calc.fastTwoSum(sh, eh + sl);
                                    [obj.hi, obj.lo] = Calc.fastTwoSum(sh, eh + el);
                                else
                                    [obj.hi, obj.lo] = dd_a_dd(a.hi,a.lo,b.hi,b.lo,threadNum,avx);
                                end
                            elseif acc == 1
                                [obj.hi, obj.lo] = dd_a_dd_ieee(a.hi,a.lo,b.hi,b.lo,threadNum,avx);
                            end
                        else
                            %obj2 is not supported
                        end
                    else
                        %obj1 is not supported
                    end
                end
            else
                disp("size error");
            end
            
        end
        %minus DD <- A - B
        function obj = minus(a, b)
            obj=plus(a,-b);
        end
        %uminus DD <- -A
        function obj = uminus(a)
            obj = DD(0);
            obj.hi = -a.hi;
            obj.lo = -a.lo;
        end
        %uplus DD <- +A
        function obj = uplus(a)
            obj = DD(0);
            obj.hi = a.hi;
            obj.lo = a.lo;
        end
        %times DD <- A.*B
        function obj = times(a, b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            
            if(m1 ~= m2 || n1 ~= n2)
                %unsupported operation
                abort();
            end
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            
            obj = DD(zeros(m1,n1));
            
            if(isD1)
                if(isDD2)
                    warning('off','MATLAB:structOnObject');
                    b=struct(b);
                    for i = 1:m1
                        for j = 1:n1
                            %pattarn double * DD
                            [p, e] = Calc.twoProd(a(i, j), b.hi(i, j));
                            
                            e = e + a(i, j) * b.lo(i, j);
                            
                            [p, e] = Calc.fastTwoSum(p, e);
                            
                            obj.hi(i, j) = p;
                            obj.lo(i, j) = e;
                        end
                    end
                else
                    %obj2 is not supported
                end
                
            else
                
                if(isDD1)
                    if(isD2)
                        warning('off','MATLAB:structOnObject');
                        a=struct(a);
                        for i = 1:m1
                            for j = 1:n1
                                %pattarn DD * double
                                [p, e] = Calc.twoProd(a.hi(i, j), b(i,j));
                                
                                e = e + a.lo(i, j) * b(i, j);
                                
                                [p, e] = Calc.fastTwoSum(p, e);
                                
                                obj.hi(i, j) = p;
                                obj.lo(i, j) = e;
                            end
                        end
                    elseif(isDD2)
                        warning('off','MATLAB:structOnObject');
                        a=struct(a);
                        b=struct(b);
                        for i = 1:m1
                            for j = 1:n1
                                %pattarn DD * DD
                                [p, e] = Calc.twoProd(a.hi(i, j), b.hi(i, j));
                                
                                e = e + a.hi(i, j) * b.lo(i, j);
                                e = e + a.lo(i, j) * b.hi(i, j);
                                
                                [p, e] = Calc.fastTwoSum(p, e);
                                
                                obj.hi(i, j) = p;
                                obj.lo(i, j) = e;
                            end
                        end
                    else
                        %obj2 is not supported
                    end
                else
                    %obj1 is not supported
                end
            end
        end
        
        %mtimes DD <- A*B
        function obj = mtimes(a, b)
            global deffma;
            global defavx;
            global defthreadNum;
            global defacc;
            acc = defacc;
            fma = deffma;
            avx = defavx;
            threadNum = defthreadNum;
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            
            if(n1 == m2)
                obj = DD(zeros(m1, n2));
            end
            
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            if(isD1)
                if(isDD2)
                    %pattarn double * DD
                    if((m1 == 1 && n1 == 1) && (m2 == 1 && n2 == 1))
                        [p, e] = Calc.twoProd(a, b.hi);
                        
                        e = e + a * b.lo;
                        
                        [p, e] = Calc.fastTwoSum(p, e);
                        
                        obj.hi = p;
                        obj.lo = e;
                    elseif((m1 == 1 && n1 == 1)) %scalar * vec or mat
                        obj = DD(zeros(m2, n2));
                        if (m2 == 1)
                            b = b';
                            [obj.hi, obj.lo] = d_scl_dd(a,b.hi,b.lo,threadNum,avx,fma);
                            obj = obj';
                        else
                            [obj.hi, obj.lo] = d_scl_dd(a,b.hi,b.lo,threadNum,avx,fma);
                        end
                    elseif(m2==1&&n2==1) %vec or mat * scalar
                        obj = DD(zeros(m1, n1));
                        [obj.hi, obj.lo] = d_scl_dd(a,b.hi,b.lo,threadNum,avx,fma);
                        if (m1 == 1)
                            b = b';
                            [obj.hi, obj.lo] = d_scl_dd(a,b.hi,b.lo,threadNum,avx,fma);
                            obj = obj';
                        else
                            [obj.hi, obj.lo] = d_scl_dd(a,b.hi,b.lo,threadNum,avx,fma);
                        end
                    elseif(n1 == m2)
                        if (n2==1&&m1==1)
                            a=a';
                            obj = DD(zeros(m1, n2));
                            [obj.hi,obj.lo] = d_dot_dd(a,b.hi,b.lo,threadNum,avx,fma);
                        elseif (n2==1)
                            %matvec
                            obj = DD(zeros(m1, n2));
                            [obj.hi, obj.lo] = d_mv_dd(a,b.hi,b.lo,threadNum,avx,fma);
                        elseif(m1==1) %1-m * m-n
                            a=a';
                            obj = DD(zeros(n1, m1));
                            [obj.hi,obj.lo] = dd_tmv_d(b.hi,b.lo,a,threadNum,avx,fma);
                            obj=obj';
                        else
                            %mat-mat
                            obj = DD(zeros(m1, n2));
                            [obj.hi, obj.lo] = d_mm_dd(a,b.hi,b.lo,threadNum,avx,fma);
                        end
                    end
                else
                    %d*d
                end
            else
                if(isDD1)
                    if(isD2)
                        %pattarn DD * double
                        if((m1 == 1 && n1 == 1) && (m2 == 1 && n2 == 1))
                            [p, e] = Calc.twoProd(a.hi, b);
                            
                            e = e + a.lo * b;
                            
                            [p, e] = Calc.fastTwoSum(p, e);
                            
                            obj.hi = p;
                            obj.lo = e;
                        elseif(m1 == 1 && n1 == 1)
                            obj = DD(zeros(m2, n2));
                            if (m2 == 1)
                                b = b';
                                [obj.hi, obj.lo] = dd_scl_d(a.hi,a.lo,b,threadNum,avx,fma);
                                obj = obj';
                            else
                                [obj.hi, obj.lo] = dd_scl_d(a.hi,a.lo,b,threadNum,avx,fma);
                            end
                        elseif(m2==1&&n2==1)
                            obj = DD(zeros(m1, n1));
                            if (m1 == 1)
                                b = b';
                                [obj.hi, obj.lo] = dd_scl_d(a.hi,a.lo,b,threadNum,avx,fma);
                                obj = obj';
                            else
                                [obj.hi, obj.lo] = dd_scl_d(a.hi,a.lo,b,threadNum,avx,fma);
                            end
                        elseif(n1 == m2)
                            if (n2==1&&m1==1)
                                a=a';
                                obj = DD(zeros(m1, n2));
                                [obj.hi,obj.lo] = d_dot_dd(b,a.hi,a.lo,threadNum,avx,fma);
                            elseif (n2==1)%mv
                                obj = DD(zeros(m1, n2));
                                [obj.hi, obj.lo] = dd_mv_d(a.hi,a.lo,b,threadNum,avx,fma);
                            elseif (m1==1)%tmv
                                a=a';
                                obj = DD(zeros(n1, m1));
                                [obj.hi,obj.lo] = d_tmv_dd(b,a.hi,a.lo,threadNum,avx,fma);
                                obj=obj';
                            else
                                %mat-mat
                                obj = DD(zeros(m1, n2));
                                [obj.hi, obj.lo] = dd_mm_d(a.hi,a.lo,b,threadNum,avx,fma);
                            end
                        end
                    elseif(isDD2)
                        %pattarn DD * DD
                        if((m1 == 1 && n1 == 1) && (m2 == 1 && n2 == 1))
                            [p, e] = Calc.twoProd(a.hi, b.hi);
                            
                            e = e + a.hi * b.lo;
                            e = e + a.lo * b.hi;
                            
                            [p, e] = Calc.fastTwoSum(p, e);
                            
                            obj.hi = p;
                            obj.lo = e;
                        elseif(m1==1&&n1==1)%ax
                            obj = DD(zeros(m2, n2));
                            if (m2 == 1)
                                b = b';
                                [obj.hi,obj.lo] = dd_scl_dd(a.hi,a.lo,b.hi,b.lo,threadNum,avx,fma);
                                obj = obj';
                            else
                                [obj.hi,obj.lo] = dd_scl_dd(a.hi,a.lo,b.hi,b.lo,threadNum,avx,fma);
                            end
                        elseif(m2==1&&n2==1)%xa
                            obj = DD(zeros(m1, n1));
                            if (m1 == 1)
                                b = b';
                                [obj.hi,obj.lo] = dd_scl_dd(a.hi,a.lo,b.hi,b.lo,threadNum,avx,fma);
                                obj = obj';
                            else
                                [obj.hi,obj.lo] = dd_scl_dd(a.hi,a.lo,b.hi,b.lo,threadNum,avx,fma);
                            end
                            
                        elseif(n1 == m2)
                            if acc == 0
                                if (n2==1&&m1==1)
                                    a=a';
                                    obj = DD(zeros(m1, n2));
                                    [obj.hi,obj.lo] = dd_dot_dd(a.hi,a.lo,b.hi,b.lo,threadNum,avx,fma);
                                elseif (n2==1)
                                    obj = DD(zeros(m1, n2));
                                    [obj.hi,obj.lo] = dd_mv_dd(a.hi,a.lo,b.hi,b.lo,threadNum,avx,fma);
                                elseif(m1==1)%1,n * n,n
                                    a=a';
                                    obj = DD(zeros(n1, m1));
                                    [obj.hi,obj.lo] = dd_tmv_dd(b.hi,b.lo,a.hi,a.lo,threadNum,avx,fma);
                                    obj=obj';
                                else
                                    obj = DD(zeros(m1, n2));
                                    [obj.hi,obj.lo] = dd_mm_dd(a.hi,a.lo,b.hi,b.lo,threadNum,avx,fma);
                                end
                            elseif acc == 1
                                if (n2==1&&m1==1)
                                    a=a';
                                    obj = DD(zeros(m1, n2));
                                    [obj.hi,obj.lo] = dd_dot_dd_ieee(a.hi,a.lo,b.hi,b.lo,threadNum,avx,fma);
                                elseif (n2==1)
                                    obj = DD(zeros(m1, n2));
                                    [obj.hi,obj.lo] = dd_mv_dd_ieee(a.hi,a.lo,b.hi,b.lo,threadNum,avx,fma);
                                elseif(m1==1)%1,n * n,n
                                    a=a';
                                    obj = DD(zeros(n1, m1));
                                    [obj.hi,obj.lo] = dd_tmv_dd_ieee(b.hi,b.lo,a.hi,a.lo,threadNum,avx,fma);
                                    obj=obj';
                                else
                                    obj = DD(zeros(m1, n2));
                                    [obj.hi,obj.lo] = dd_mm_dd_ieee(a.hi,a.lo,b.hi,b.lo,threadNum,avx,fma);
                                end
                            end
                        end
                    end
                end
            end
        end
        %rdivide DD <- A./B
        function obj = rdivide(a, b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            if(m1 ~= m2 || n1 ~= n2)
                %unsupported operation
                abort();
            end
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            obj = DD(zeros(m1,n1));
            for i = 1:m1
                for j = 1:n1
                    if(isD1)
                        if(isDD2)
                            % d / DD
                            c = a(i, j) / b.hi(i, j);
                            [p, e] = Calc.twoProd(c, b.hi(i, j));
                            
                            cc = (a(i, j) - p - e - c*b.lo(i, j)) / b.hi(i, j);
                            
                            [obj.hi(i, j), obj(i, j).lo] = Calc.fastTwoSum(c, cc);
                            
                        else
                            %unsupported b type
                        end
                    elseif(isDD1)
                        if(isD2)
                            % DD / d
                            c = a.hi(i, j) / b(i, j);
                            [p, e] = Calc.twoProd(c, b(i, j));
                            
                            cc = (a.hi(i, j) - p - e + a.lo(i, j)) / b(i, j);
                            
                            [obj.hi(i, j), obj.lo(i, j)] = Calc.fastTwoSum(c, cc);
                        elseif(isDD2)
                            % DD / DD
                            c = a.hi(i, j) / b.hi(i, j);
                            [p, e] = Calc.twoProd(c, b.hi(i, j));
                            
                            cc = (a.hi(i, j) - p - e + a.lo(i, j) - c*b.lo(i, j)) / b.hi(i, j);
                            
                            [obj.hi(i, j), obj.lo(i, j)] = Calc.fastTwoSum(c, cc);
                        else
                            %unsupported b type
                        end
                    else
                        %unsupported a type
                    end
                end
            end
        end
        %ldivide DD <- A.\B
        function obj = ldivide(a, b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            if(m1 ~= m2 || n1 ~= n2)
                %unsupported operation
                abort();
            end
            
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            obj = DD(zeros(m1,n1));
            for i = 1:m1
                for j = 1:n1
                    if(isD1)
                        if(isDD2)
                            % d / DD
                            c = b(i, j) / a.hi(i, j);
                            [p, e] = Calc.twoProd(c, a.hi(i, j));
                            
                            cc = (b(i, j) - p - e - c*a.lo(i, j)) / a.hi(i, j);
                            
                            [obj.hi(i, j), obj(i, j).lo] = Calc.fastTwoSum(c, cc);
                            
                        else
                            %unsupported b type
                        end
                    elseif(isDD1)
                        if(isD2)
                            % DD / d
                            c = b.hi(i, j) / a(i, j);
                            [p, e] = Calc.twoProd(c, a(i, j));
                            
                            cc = (b.hi(i, j) - p - e + b.lo(i, j)) / a(i, j);
                            
                            [obj.hi(i, j), obj.lo(i, j)] = Calc.fastTwoSum(c, cc);
                        elseif(isDD2)
                            % DD / DD
                            c = b.hi(i, j) / a.hi(i, j);
                            [p, e] = Calc.twoProd(c, a.hi(i, j));
                            
                            cc = (b.hi(i, j) - p - e + b.lo(i, j) - c*a.lo(i, j)) / a.hi(i, j);
                            
                            [obj.hi(i, j), obj.lo(i, j)] = Calc.fastTwoSum(c, cc);
                        else
                            %unsupported b type
                        end
                    else
                        %unsupported a type
                    end
                end
            end
        end
        %mrdivide DD <- A/B
        function obj = mrdivide(a, b)
            [m2, n2] = size(b);
            if(m2 ~= 1 || n2 ~= 1)
                %unsupported operation
                abort();
            end
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            obj = DD(0);
            if(isD1)
                if(isDD2)
                    % d / DD
                    c = a / b.hi;
                    [p, e] = Calc.twoProd(c, b.hi);
                    
                    cc = (a - p - e - c*b.lo) / b.hi;
                    
                    [obj.hi, obj.lo] = Calc.fastTwoSum(c, cc);
                    
                else
                    %unsupported b type
                end
            elseif(isDD1)
                if(isD2)
                    % DD / d
                    c = a.hi / b;
                    [p, e] = Calc.twoProd(c, b);
                    
                    cc = (a.hi - p - e + a.lo) / b;
                    
                    [obj.hi, obj.lo] = Calc.fastTwoSum(c, cc);
                elseif(isDD2)
                    % DD / DD
                    c = a.hi / b.hi;
                    [p, e] = Calc.twoProd(c, b.hi);
                    
                    cc = (a.hi - p - e + a.lo - c*b.lo) / b.hi;
                    
                    [obj.hi, obj.lo] = Calc.fastTwoSum(c, cc);
                else
                    %unsupported b type
                end
            else
                %unsupported a type
            end
        end
        %mldivide DD <- A\B
        function obj = mldivide(a, b)
            [m1, n1] = size(a);
            if(m1 ~= 1 && n1 ~= 1)
                %unsupported operation
                abort();
            end
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            if(isD2)
                if(isDD1)
                    % d / DD
                    c = b / a.hi;
                    [p, e] = Calc.twoProd(c, a.hi);
                    
                    cc = (b - p - e - c*a.lo) / a.hi;
                    
                    [obj.hi, obj.lo] = Calc.fastTwoSum(c, cc);
                    
                else
                    %unsupported a type
                end
            elseif(isDD2)
                if(isD1)
                    % DD / d
                    c = b.hi / a;
                    [p, e] = Calc.twoProd(c, a);
                    
                    cc = (b.hi - p - e + b.lo) / a;
                    
                    [obj.hi, obj.lo] = Calc.fastTwoSum(c, cc);
                elseif(isDD1)
                    % DD / DD
                    c = b.hi / a.hi;
                    [p, e] = Calc.twoProd(c, a.hi);
                    
                    cc = (b.hi - p - e + b.lo - c*a.lo) / a.hi;
                    
                    [obj.hi, obj.lo] = Calc.fastTwoSum(c, cc);
                else
                    %unsupported a type
                end
            else
                %unsupported b type
            end
            
        end
        %lt bool <- a < b
        function obj = lt(a, b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            isScalarA = false;
            isScalarB = false;
            % size(a) = [1,1]
            if(m1 == 1 && n1 == 1)
                isScalarA = true;
            end
            if(m2 == 1 && n2 == 1)
                isScalarB = true;
            end
            if((m1 == m2 && n1 == n2) || (isScalarA || isScalarB))
                %unsupported oparation
            end
            
            if(isScalarA)
                m1 = m2;
                n1 = n2;
            end
            
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            if(isD1)
                if(isDD2)
                    % d < DD
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a(i, j) < b.hi(i, j) ||(a(i, j) == b.hi(i, j) && 0 < b.lo(i, j)))
                                obj(i, j) = true;
                            end
                        end
                    end
                else
                    %unsupported b type
                end
            elseif(isDD1)
                if(isD2)
                    %DD < d
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a.hi(i, j) < b(i, j) || (a.hi(i, j) == b(i, j) && 0 < b(i, j)))
                                obj(i, j) = true;
                            end
                        end
                    end
                elseif(isDD2)
                    % DD < DD
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a.hi(i, j) < b.hi(i, j) || (a.hi(i, j) == b.hi(i, j) && a.lo(i,j) < b.lo(i, j)))
                                obj(i, j) = true;
                            end
                        end
                    end
                else
                    %unsupported b type
                end
            else
                %unsupported a type
            end
            obj=logical(obj);
        end
        %gt bool <- a > b
        function obj = gt(a, b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            isScalarA = false;
            isScalarB = false;
            % size(a) = [1,1]
            if(m1 == 1 && n1 == 1)
                isScalarA = true;
            end
            if(m2 == 1 && n2 == 1)
                isScalarB = true;
            end
            if((m1 == m2 && n1 == n2) || (isScalarA || isScalarB))
                %unsupported oparation
            end
            
            if(isScalarA)
                m1 = m2;
                n1 = n2;
            end
            
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            if(isD1)
                if(isDD2)
                    % d < DD
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a(i, j) > b.hi(i, j) ||(a(i, j) == b.hi(i, j) && 0 > b.lo(i, j)))
                                obj(i, j) = true;
                            end
                        end
                    end
                else
                    %unsupported b type
                end
            elseif(isDD1)
                if(isD2)
                    %DD < d
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a.hi(i, j) > b(i, j) ||...
                                    (a.hi(i, j) == b(i, j) && 0 > b(i, j)))
                                obj(i, j) = true;
                            end
                        end
                    end
                elseif(isDD2)
                    % DD < DD
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a.hi(i, j) > b.hi(i, j) ||...
                                    (a.hi(i, j) == b.hi(i, j) && a.lo(i,j) > b.lo(i, j)))
                                obj(i, j) = true;
                            end
                        end
                    end
                else
                    %unsupported b type
                end
            else
                %unsupported a type
            end
            obj=logical(obj);
        end
        %lt bool <- a <= b
        function obj = le(a, b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            isScalarA = false;
            isScalarB = false;
            % size(a) = [1,1]
            if(m1 == 1 && n1 == 1)
                isScalarA = true;
            end
            if(m2 == 1 && n2 == 1)
                isScalarB = true;
            end
            if((m1 == m2 && n1 == n2) || (isScalarA || isScalarB))
                %unsupported oparation
            end
            
            if(isScalarA)
                m1 = m2;
                n1 = n2;
            end
            
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            if(isD1)
                if(isDD2)
                    % d < DD
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a(i, j) < b.hi(i, j) ||(a(i, j) == b.hi(i, j) && 0 <= b.lo(i, j)))
                                obj(i, j) = true;
                            end
                        end
                    end
                else
                    %unsupported b type
                end
            elseif(isDD1)
                if(isD2)
                    %DD < d
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a.hi(i, j) < b(i, j) ||...
                                    (a.hi(i, j) == b(i, j) && 0 <= b(i, j)))
                                obj(i, j) = true;
                            end
                        end
                    end
                elseif(isDD2)
                    % DD < DD
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a.hi(i, j) < b.hi(i, j) ||...
                                    (a.hi(i, j) == b.hi(i, j) && a.lo(i,j) <= b.lo(i, j)))
                                obj(i, j) = true;
                            end
                        end
                    end
                else
                    %unsupported b type
                end
            else
                %unsupported a type
            end
            obj=logical(obj);
        end
        %lt bool <- a => b
        function obj = ge(a, b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            isScalarA = false;
            isScalarB = false;
            % size(a) = [1,1]
            if(m1 == 1 && n1 == 1)
                isScalarA = true;
            end
            if(m2 == 1 && n2 == 1)
                isScalarB = true;
            end
            if((m1 == m2 && n1 == n2) || (isScalarA || isScalarB))
                %unsupported oparation
            end
            
            if(isScalarA)
                m1 = m2;
                n1 = n2;
            end
            
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            if(isD1)
                if(isDD2)
                    % d < DD
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a(i, j) > b.hi(i, j) ||(a(i, j) == b.hi(i, j) && 0 >= b.lo(i, j)))
                                obj(i, j) = true;
                            end
                        end
                    end
                else
                    %unsupported b type
                end
            elseif(isDD1)
                if(isD2)
                    %DD < d
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a.hi(i, j) > b(i, j) ||...
                                    (a.hi(i, j) == b(i, j) && 0 >= b(i, j)))
                                obj(i, j) = true;
                            end
                        end
                    end
                elseif(isDD2)
                    % DD < DD
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a.hi(i, j) > b.hi(i, j) ||...
                                    (a.hi(i, j) == b.hi(i, j) && a.lo(i,j) >= b.lo(i, j)))
                                obj(i, j) = true;
                            end
                        end
                    end
                else
                    %unsupported b type
                end
            else
                %unsupported a type
            end
            obj=logical(obj);
        end
        %lt bool <- a ~= b
        function obj = ne(a, b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            isScalarA = false;
            isScalarB = false;
            % size(a) = [1,1]
            if(m1 == 1 && n1 == 1)
                isScalarA = true;
            end
            if(m2 == 1 && n2 == 1)
                isScalarB = true;
            end
            if((m1 == m2 && n1 == n2) || (isScalarA || isScalarB))
                %unsupported oparation
            end
            
            if(isScalarA)
                m1 = m2;
                n1 = n2;
            end
            
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            if(isD1)
                if(isDD2)
                    % d < DD
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a(i, j) ~= b.hi(i, j) || b.lo(i, j) ~= 0)
                                obj(i, j) = true;
                            end
                        end
                    end
                else
                    %unsupported b type
                end
            elseif(isDD1)
                if(isD2)
                    %DD < d
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a.hi(i, j) ~= b(i, j) || b(i, j) ~= 0)
                                obj(i, j) = true;
                            end
                        end
                    end
                elseif(isDD2)
                    % DD < DD
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a.hi(i, j) ~= b.hi(i, j) || a.lo(i,j) ~= b.lo(i, j))
                                obj(i, j) = true;
                            end
                        end
                    end
                else
                    %unsupported b type
                end
            else
                %unsupported a type
            end
            obj=logical(obj);
        end
        %eq bool <- a == b
        function obj = eq(a, b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            isScalarA = false;
            isScalarB = false;
            % size(a) = [1,1]
            if(m1 == 1 && n1 == 1)
                isScalarA = true;
            end
            if(m2 == 1 && n2 == 1)
                isScalarB = true;
            end
            if((m1 == m2 && n1 == n2) || (isScalarA || isScalarB))
                %unsupported oparation
            end
            
            if(isScalarA)
                m1 = m2;
                n1 = n2;
            end
            
            isD1 = isnumeric(a);
            isD2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            if(isD1)
                if(isDD2)
                    % d < DD
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a(i, j) == b.hi(i, j) && b.lo(i, j) == 0)
                                obj(i, j) = true;
                            end
                        end
                    end
                else
                    %unsupported b type
                end
            elseif(isDD1)
                if(isD2)
                    %DD < d
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a.hi(i, j) == b(i, j) && b(i, j) == 0)
                                obj(i, j) = true;
                            end
                        end
                    end
                elseif(isDD2)
                    % DD < DD
                    obj = zeros(m1, n1);
                    for i = 1:m1
                        for j = 1:n1
                            if(a.hi(i, j) == b.hi(i, j) && a.lo(i,j) == b.lo(i, j))
                                obj(i, j) = true;
                            end
                        end
                    end
                else
                    %unsupported b type
                end
            else
                %unsupported a type
            end
            obj=logical(obj);
        end
        %and bool <- a & b
        function obj = and(a, b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            if(m1 ~= m2 ||  n1 ~= n2)
                %unsupported operation
            end
            obj = double(a) & double(b);
        end
        %or bool <- a | b
        function obj = or(a, b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            if(m1 ~= m2 ||  n1 ~= n2)
                %unsupported operation
            end
            obj = double(a) | double(b);
        end
        %not bool <- ~a
        function obj = not(a)
            obj = ~double(a);
        end
        %colon obj <- a:b
        %ctransepose obj <- a.'
        function obj = ctranspose(a)
            obj = DD(0);
            obj.hi = a.hi';
            obj.lo = a.lo';
        end
        %transepose obj <- a.'
        function obj = transpose(a)
            obj = DD(0);
            obj.hi = a.hi.';
            obj.lo = a.lo.';
        end
        %horzcat obj <- [a b ...]
        function obj = horzcat(varargin)
            obj = DD(0);
            if(isnumeric(varargin{1}))
                [m, n] = size(varargin{1});
                obj.hi = varargin{1};
                obj.lo = zeros(m, n);
            elseif(isa(varargin{1}, 'DD'))
                obj.hi = varargin{1}.hi;
                obj.lo = varargin{1}.lo;
            else
                %unsupported operation
            end
            for i = 2:length(varargin)
                if(isnumeric(varargin{i}))
                    %pattarn double
                    [m, n] = size(varargin{i});
                    obj.hi = [obj.hi varargin{i}];
                    obj.lo = [obj.lo zeros(m, n)];
                elseif(isa(varargin{i}, 'DD'))
                    %pattarn DD
                    obj.hi = [obj.hi varargin{i}.hi];
                    obj.lo = [obj.lo varargin{i}.lo];
                else
                    %unsuppoted oparation
                end
            end
            
        end
        %vertcat obj <- [a;b; ...]
        function obj = vertcat(varargin)
            obj = DD(0);
            if(isnumeric(varargin{1}))
                [m, n] = size(varargin{1});
                obj.hi = varargin{1};
                obj.lo = zeros(m, n);
            elseif(isa(varargin{1}, 'DD'))
                obj.hi = varargin{1}.hi;
                obj.lo = varargin{1}.lo;
            else
                %unsupported operation
            end
            for i = 2:length(varargin)
                if(isnumeric(varargin{i}))
                    %pattarn double
                    [m, n] = size(varargin{i});
                    obj.hi = [obj.hi;varargin{i}];
                    obj.lo = [obj.lo;zeros(m, n)];
                elseif(isa(varargin{i}, 'DD'))
                    %pattarn DD
                    obj.hi = [obj.hi;varargin{i}.hi];
                    obj.lo = [obj.lo;varargin{i}.lo];
                else
                    %unsuppoted oparation
                end
            end
            
        end
        %subsref obj <- obj(i, j)
        function obj = subsref(A, S)
            switch S.type
                case '()'
                    %A(n)
                    obj = DD(0);
                    obj.hi = A.hi(S.subs{:});
                    obj.lo = A.lo(S.subs{:});
                case '{}'
                    %A{n}
                case '.'
                    %A.n
                    obj = builtin('subsref',A,S);
                    
                    
            end
        end
        %subsasgn obj <- obj = B
        function obj = subsasgn(obj, S, B)
            switch S.type
                case '()'
                    if(isnumeric(B))
                        tmp = DD(B);
                        obj.hi(S.subs{:}) = tmp.hi;
                        obj.lo(S.subs{:}) = tmp.lo;
                    end
                    if(isa(B,'DD'))
                        obj.hi(S.subs{:}) = B.hi;
                        obj.lo(S.subs{:}) = B.lo;
                    end
                    if(isa(B,'QD'))
                        disp("error:");
                        disp("You should cast DD to QD.");
                    end
                case '{}'
                case '.'
                    %obj = builtin('subsref',B,S);
            end
        end
        
        
    end
    
end
