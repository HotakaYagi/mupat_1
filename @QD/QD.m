classdef(InferiorClasses = ?DD) QD<matlab.mixin.CustomDisplay
    
    properties(GetAccess = public, SetAccess = private,Hidden = false)
        %set parameter of QD
        hh=0;
        hl=0;
        lh=0;
        ll=0;
    end
    
    methods (Static)
        function out = qdformat(varargin)
            persistent style
            if(isempty(style))
                style = 0;
            end
            
            if(isempty(varargin))
                out=style;
            elseif(length(varargin)==1)
                style = varargin{1};
            end
        end
    end
    
    methods (Access = protected)
        function propgrp = getPropertyGroups(obj)
            switch(QD.qdformat())
                case 0
                    propList = struct('hh',obj.hh,'hl',obj.hl,'lh',obj.lh,'ll',obj.ll);
                    propgrp = matlab.mixin.util.PropertyGroup(propList);
                case 1
                    [m, n] = size(obj.hh);
                    if(m == 1 && n == 1)
                        propList = struct('QD',qdprint(obj));
                        propgrp = matlab.mixin.util.PropertyGroup(propList);
                    elseif(m == 1 || n == 1)
                        propList = struct('QD',strcat('[',num2str(m),'×', num2str(n), '  vector ]') );
                        propgrp = matlab.mixin.util.PropertyGroup(propList);
                    else
                        propList = struct('QD',strcat('[',num2str(m),'×', num2str(n), '  matrix ]') );
                        propgrp = matlab.mixin.util.PropertyGroup(propList);
                    end
            end
        end
        function header = getHeader(obj)
            header = sprintf('');
        end
    end
    
    methods
        function obj = QD(a,b,c,d)
            if nargin==1
                if isnumeric(a)==1
                    [n_a,m_a] = size(a);
                    obj.hh = a;
                    obj.hl = zeros(n_a, m_a);
                    obj.lh = zeros(n_a, m_a);
                    obj.ll = zeros(n_a, m_a);
                elseif isa(a,'DD')==1
                    [n_a,m_a] = size(a.hi);
                    obj.hh = a.hi;
                    obj.hl = a.lo;
                    obj.lh = zeros(n_a, m_a);
                    obj.ll = zeros(n_a, m_a);
                elseif isa(a,'QD')==1
                    obj.hh = a.hh;
                    obj.hl = a.hl;
                    obj.lh = a.lh;
                    obj.ll = a.ll;
                end
            elseif(nargin==2)
                obj.hh = a;
                obj.hl = b;
            elseif nargin == 3
                obj.hh = a;
                obj.hl = b;
                obj.lh = c;
            elseif(nargin==4)
                obj.hh = a;
                obj.hl = b;
                obj.lh = c;
                obj.ll = d;
            elseif(nargin==0)
                obj.hh = 0;
                obj.hl = 0;
                obj.lh = 0;
                obj.ll = 0;
            else
                disp('Error: arguments is 0,1,2,4');
            end
        end
        
        function obj = qd(a, b, c, d)
            obj = QD(a, b, c, d);
        end
        %-----------------%
        %   base method   %
        %-----------------%
        
        function obj = transpose(a)
            obj = QD(0);
            obj.hh = a.hh.';
            obj.hl = a.hl.';
            obj.lh = a.lh.';
            obj.ll = a.ll.';
        end
        
        function obj = ctranspose(a)
            obj = QD(0);
            obj.hh = a.hh';
            obj.hl = a.hl';
            obj.lh = a.lh';
            obj.ll = a.ll';
        end
        
        function [n,m]=size(obj)
            [n,m]=size(obj.hh);
        end
        
        function d=double(obj)
            d=obj.hh;
        end
        
        function d=Double(obj)
            d=obj.hh;
        end
        
        function obj=sparse(A)
            obj=QD(sparse(A.hh),sparse(A.hl),sparse(A.lh),sparse(A.ll));
        end
        
        function obj=full(A)
            obj=QD(full(A.hh),full(A.hl),full(A.lh),full(A.ll));
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
        
        function obj=abs(a)
            obj=a;
            c = find(sign(a.hh) == -1);
            obj.hh(c) = (-1) * a.hh(c);
            obj.hl(c) = (-1) * a.hl(c);
            obj.lh(c) = (-1) * a.lh(c);
            obj.ll(c) = (-1) * a.ll(c);
        end
        
        function obj=sqrt(a)
            [m,n]=size(a);
            obj=QD(zeros(m,n));
            warning('off','MATLAB:structOnObject');
            a=struct(a);
            for i=1:m
                for j=1:n
                    if a.hh(i,j)==0&&a.hl(i,j)==0&&a.lh(i,j)==0&&a.ll(i,j)==0
                        obj.hh(i,j)=0;
                        obj.hl(i,j)=0;
                        obj.lh(i,j)=0;
                        obj.ll(i,j)=0;
                        continue
                    end
                    c = 1.0/sqrt(a.hh(i,j));
                    c = QD(c);
                    h = 0.5*QD(a.hh(i,j),a.hl(i,j),a.lh(i,j),a.ll(i,j));
                    c = c + (0.5 - h*(c*c))*c;
                    c = c + (0.5 - h*(c*c))*c;
                    c = c + (0.5 - h*(c*c))*c;
                    c = c*QD(a.hh(i,j),a.hl(i,j),a.lh(i,j),a.ll(i,j));
                    obj.hh(i,j)=c.hh;
                    obj.hl(i,j)=c.hl;
                    obj.lh(i,j)=c.lh;
                    obj.ll(i,j)=c.ll;
                end
            end
        end
        
        %Inner prod
        function z=dot(x,y)
            global deffma;
            global defavx;
            global defthreadNum;
            fma = deffma;
            avx = defavx;
            threadNum = defthreadNum;
            z=QD(0);
            isxd = isnumeric(x);
            isyd = isnumeric(y);
            isxdd = isa(x, 'DD');
            isydd = isa(y, 'DD');
            isxqd = isa(x, 'QD');
            isyqd = isa(y, 'QD');
            if(isxd)
                if(isyqd)
                    [z.hh,z.hl,z.lh,z.ll]=d_dot_qd(x,y.hh,y.hl,y.lh,y.ll,threadNum,avx,fma);
                end
            elseif(isxdd)
                if(isyqd)
                    [z.hh,z.hl,z.lh,z.ll]=dd_dot_qd(x.hi,x.lo,y.hh,y.hl,y.lh,y.ll,threadNum,avx,fma);
                end
            elseif(isxqd)
                if(isyd)
                    [z.hh,z.hl,z.lh,z.ll]=d_dot_qd(y,x.hh,x.hl,x.lh,x.ll,threadNum,avx,fma);
                elseif(isydd)
                    [z.hh,z.hl,z.lh,z.ll]=dd_dot_qd(y.hi,y.lo,x.hh,x.hl,x.lh,x.ll,threadNum,avx,fma);
                elseif(isyqd)
                    [z.hh,z.hl,z.lh,z.ll]=qd_dot_qd(x.hh,x.hl,x.lh,x.ll,y.hh,y.hl,y.lh,y.ll,threadNum,avx,fma);
                end
            end
        end
        
        %transposed matrix vector multiplication
        function z=tmv(x,y)
            global deffma;
            global defavx;
            global defthreadNum;
            fma = deffma;
            avx = defavx;
            threadNum = defthreadNum;
            z=QD(0);
            isxd = isnumeric(x);
            isyd = isnumeric(y);
            isxdd = isa(x, 'DD');
            isydd = isa(y, 'DD');
            isxqd = isa(x, 'QD');
            isyqd = isa(y, 'QD');
            if(isxd)
                if(isyqd)
                    [z.hh,z.hl,z.lh,z.ll]=d_tmv_qd(x,y.hh,y.hl,y.lh,y.ll,threadNum,avx,fma);
                end
            elseif(isxdd)
                if(isyqd)
                    [z.hh,z.hl,z.lh,z.ll]=dd_tmv_qd(x.hi,x.lo,y.hh,y.hl,y.lh,y.ll,threadNum,avx,fma);
                end
            elseif(isxqd)
                if(isyd)
                    [z.hh,z.hl,z.lh,z.ll]=qd_tmv_d(x.hh,x.hl,x.lh,x.ll,y,threadNum,avx,fma);
                elseif(isydd)
                    [z.hh,z.hl,z.lh,z.ll]=qd_tmv_dd(x.hh,x.hl,x.lh,x.ll,y.hi,y.lo,threadNum,avx,fma);
                elseif(isyqd)
                    [z.hh,z.hl,z.lh,z.ll]=qd_tmv_qd(x.hh,x.hl,x.lh,x.ll,y.hh,y.hl,y.lh,y.ll,threadNum,avx,fma);
                end
            end
        end
        
        function obj=norm(x,N)
            [n,m]=size(x);
            if m~=1
                disp('set vector');
            end
            switch(N)
                case 1
                    obj=QD(0);
                    x=abs(x);
                    x=struct(x);
                    for i=1:n
                        obj=obj+QD(x.hh(i),x.hl(i),x.lh(i),x.ll(i));
                    end
                case 2
                    obj = sqrt(dot(x,x));
                case 'inf'
                    x=abs(x);
                    x=struct(x);
                    x_max=QD(x.hh(1),x.hl(1),x.lh(1),x.ll(1));
                    for i=1:n
                        if(QD(x.hh(i),x.hl(i),x.lh(i),x.ll(i)) > x_max)
                            x_max = QD(x.hh(i),x.hl(i),x.lh(i),x.ll(i));
                        end
                    end
                    obj = x_max;
                case 'fro'
                    obj=QD(0);
                    x=struct(x);
                    for i=1:n
                        for j=1:m
                            obj=obj+QD(x.hh(i,j),x.hl(i,j),x.lh(i,j),x.ll(i,j))^2;
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
                    obj=QD(0);
                elseif n>0
                    obj=qdpow(a,n);
                else
                    obj=qdpow(1/a,-n);
                end
            end
        end
        
        function p = qdpow(a,n)
            p = QD(1);
            if n==0
            else
                N = abs(n);
                if(a.hh == 0)
                    if(n > 0)
                        p = QD(0);
                    else
                        error(27) %Divide by zero...
                    end
                else
                    r=a;
                    if N>1
                        while N>0
                            if mod(N,2)==1
                                p=p*r;
                            end
                            N=fix(N/2);
                            if N>0
                                r=r*r;
                            end
                        end
                    else
                        p=r;
                    end
                    if(n<0)
                        p=1/p;
                    end
                end
            end
        end
        
        function qdprint(A)
            ln = 64;
            ib = zeros(ln,1);
            digits = ['0';'1';'2';'3';'4';'5';'6';'7';'8';'9'];
            f = QD(10);
            s = QD(0);
            [m, n] = size(A.hh);
            C = cell(m,n);
            a = QD(0);
            for x = 1:m
                for y = 1:n
                    a.hh = A.hh(x,y);
                    a.hl = A.hl(x,y);
                    a.lh = A.lh(x,y);
                    a.ll = A.ll(x,y);
                    % Determine exact power of ten for exponent.
                    if(a.hh ~= 0)
                        t1 = log10(abs(a.hh));
                        if(t1 >= 0)
                            nx = fix(t1);
                        else
                            nx = fix(t1) - 1;
                        end
                        s = a/(qdpow(f,nx));
                        if(s.hh < 0)
                            s.hh = -s.hh;
                            s.hl = -s.hl;
                            s.lh = -s.lh;
                            s.ll = -s.ll;
                        end
                        
                        % If we didn't quite get it exactly right,multiply or divide by 10 to fix.
                        i = 0;
                        
                        if((1 <= s.hh) &&( s.hh < 10))
                            bool = false;
                        else
                            bool = true;
                        end
                        
                        while((s.hh < 1) || (s.hh >= 10 )|| (bool == true))
                            i = i + 1;
                            if(s.hh < 1)
                                nx = nx - 1;
                                s = s * 10;
                                if(i > 3)
                                    bool = false;
                                end
                            elseif(s.hh >= 10)
                                nx = nx + 1;
                                s = s / 10;
                            end
                        end
                    else
                        nx = 0;
                    end
                    % compute digits
                    for i=1:ln
                        ib(i) = fix(s.hh);
                        s = (s - ib(i)) * 10;
                    end
                    % fix negative digits
                    
                    for i=ln:-1:2
                        if(ib(i) < 0)
                            ib(i) = ib(i) + 10;
                            ib(i-1) = ib(i-1) - 1;
                        end
                    end
                    
                    if(ib(1) < 0)
                        disp('qdprint : negative leading digit')
                        c = 0;
                    end
                    
                    if(ib(ln) >= 5)
                        ib(ln-1) = ib(ln-1) + 1;
                        
                        
                        for i=ln-1:-1:2
                            if(ib(i) == 10)
                                ib(i) = 0;
                                ib(i-1) = ib(i-1) + 1;
                            end
                        end
                        
                        if(ib(1) == 10)
                            ib(1) = 1;
                            nx = nx + 1;
                        end
                    end
                    
                    
                    
                    c = '';
                    if(a.hh >= 0)
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
                        d2 = fix(d1/10);
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
        
        function b=length(a)
            b=size(a);
        end
        
        function obj=triu(a,k)
            obj=QD(triu(a.hh,k),triu(a.hl,k),triu(a.lh,k),triu(a.ll,k));
        end
        
        function obj=diag(a)
            obj=QD(diag(a.hh),diag(a.hl),diag(a.lh),diag(a.ll));
        end
        %ceil
        function b=ceil(a)
            [m,n] = size(a);
            O = zeros(m,n);
            b = QD(double(ceil(DD(a))),O,O,O);
            c = find(a.hh == ceil(a.hh));
            if(size(c)~=0)
                b.hl(c) = ceil(a.hl(c));
                c = find(a.hl == ceil(a.hl));
                if(size(c)~=0)
                    b.lh(c) = ceil(a.lh(c));
                    c = find(a.lh == ceil(a.lh));
                    if(size(c)~=0)
                        b.ll(c) = ceil(a.ll(c));
                    end
                end
                [b.hh,b.hl,b.lh,b.ll] = Calc.renormalize(b.hh,b.hl,b.lh,b.ll,O);
            end
        end
        
        %floor
        function b=floor(a)
            [m,n] = size(a);
            O = zeros(m,n);
            b = QD(double(floor(DD(a))),O,O,O);
            c = find(a.hh == floor(a.hh));
            if(size(c)~=0)
                b.hl(c) = floor(a.hl(c));
                c = find(a.hl == floor(a.hl));
                if(size(c)~=0)
                    b.lh(c) = floor(a.lh(c));
                    c = find(a.lh == floor(a.lh));
                    if(size(c)~=0)
                        b.ll(c) = floor(a.ll(c));
                    end
                end
                [b.hh,b.hl,b.lh,b.ll] = Calc.renormalize(b.hh,b.hl,b.lh,b.ll,O);
            end
        end
        
        
        function obj = subsref(A, S)
            switch S.type
                case '()'
                    %A(n)
                    obj = QD(0);
                    obj.hh = A.hh(S.subs{:});
                    obj.hl = A.hl(S.subs{:});
                    obj.lh = A.lh(S.subs{:});
                    obj.ll = A.ll(S.subs{:});
                case '{}'
                    %A{n}
                case '.'
                    %A.n
                    obj = builtin('subsref',A,S);
            end
        end
        
        function obj = subsasgn(obj, S, B)
            switch S.type
                case '()'
                    if(isnumeric(B))
                        tmp = QD(B);
                        obj.hh(S.subs{:}) = tmp.hh;
                        obj.hl(S.subs{:}) = tmp.hl;
                        obj.lh(S.subs{:}) = tmp.lh;
                        obj.ll(S.subs{:}) = tmp.ll;
                    end
                    if(isa(B,'DD'))
                        tmp = QD(B);
                        obj.hh(S.subs{:}) = tmp.hh;
                        obj.hl(S.subs{:}) = tmp.hl;
                        obj.lh(S.subs{:}) = tmp.lh;
                        obj.ll(S.subs{:}) = tmp.ll;
                    end
                    if(isa(B,'QD'))
                        obj.hh(S.subs{:}) = B.hh;
                        obj.hl(S.subs{:}) = B.hl;
                        obj.lh(S.subs{:}) = B.lh;
                        obj.ll(S.subs{:}) = B.ll;
                    end
                case '{}'
                case '.'
                    %obj = builtin('subsref',B,S);
            end
        end
        
        %[a b ...]
        function obj = horzcat(varargin)
            obj = QD(0);
            if(isnumeric(varargin{1}))
                [m, n] = size(varargin{1});
                obj.hh = varargin{1};
                obj.hl = zeros(m, n);
                obj.lh = zeros(m, n);
                obj.ll = zeros(m, n);
            elseif(isa(varargin{1}, 'DD'))
                [m, n] = size(varargin{1});
                obj.hh = varargin{1}.hi;
                obj.hl = varargin{1}.lo;
                obj.lh = zeros(m, n);
                obj.ll = zeros(m, n);
            elseif(isa(varargin{1}, 'QD'))
                obj.hh = varargin{1}.hh;
                obj.hl = varargin{1}.hl;
                obj.lh = varargin{1}.lh;
                obj.ll = varargin{1}.ll;
                %unsupported operation
            end
            for i = 2:length(varargin)
                if(isnumeric(varargin{i}))
                    [m, n] = size(varargin{i});
                    obj.hh = [obj.hh varargin{i}];
                    obj.hl = [obj.hl zeros(m, n)];
                    obj.lh = [obj.lh zeros(m, n)];
                    obj.ll = [obj.ll zeros(m, n)];
                elseif(isa(varargin{i}, 'DD'))
                    [m, n] = size(varargin{i});
                    obj.hh = [obj.hh varargin{i}.hi];
                    obj.hl = [obj.hl varargin{i}.lo];
                    obj.lh = [obj.lh zeros(m, n)];
                    obj.ll = [obj.ll zeros(m, n)];
                elseif(isa(varargin{i}, 'QD'))
                    obj.hh = [obj.hh varargin{i}.hh];
                    obj.hl = [obj.hl varargin{i}.hl];
                    obj.lh = [obj.lh varargin{i}.lh];
                    obj.ll = [obj.ll varargin{i}.ll];
                else
                end
            end
        end
        %[a;b; ...]
        function obj = vertcat(varargin)
            obj = QD(0);
            if(isnumeric(varargin{1}))
                [m, n] = size(varargin{1});
                obj.hh = varargin{1};
                obj.hl = zeros(m, n);
                obj.lh = zeros(m, n);
                obj.ll = zeros(m, n);
            elseif(isa(varargin{1}, 'DD'))
                [m, n] = size(varargin{1});
                obj.hh = varargin{1}.hi;
                obj.hl = varargin{1}.lo;
                obj.lh = zeros(m, n);
                obj.ll = zeros(m, n);
            elseif(isa(varargin{1}, 'QD'))
                obj.hh = varargin{1}.hh;
                obj.hl = varargin{1}.hl;
                obj.lh = varargin{1}.lh;
                obj.ll = varargin{1}.ll;
                %unsupported operation
            end
            for i = 2:length(varargin)
                if(isnumeric(varargin{i}))
                    [m, n] = size(varargin{i});
                    obj.hh = [obj.hh; varargin{i}];
                    obj.hl = [obj.hl; zeros(m, n)];
                    obj.lh = [obj.lh; zeros(m, n)];
                    obj.ll = [obj.ll; zeros(m, n)];
                elseif(isa(varargin{i}, 'DD'))
                    [m, n] = size(varargin{i});
                    obj.hh = [obj.hh; varargin{i}.hi];
                    obj.hl = [obj.hl; varargin{i}.lo];
                    obj.lh = [obj.lh; zeros(m, n)];
                    obj.ll = [obj.ll; zeros(m, n)];
                elseif(isa(varargin{i}, 'QD'))
                    obj.hh = [obj.hh; varargin{i}.hh];
                    obj.hl = [obj.hl; varargin{i}.hl];
                    obj.lh = [obj.lh; varargin{i}.lh];
                    obj.ll = [obj.ll; varargin{i}.ll];
                else
                end
            end
        end
        
        function obj=qdeye(n,m)
            if nargin==0
                obj=QD(eye(),zeros(),zeros(),zeros());
            elseif nargin == 1
                obj=QD(eye(m),zeros(m),zeros(m),zeros(m));
            else
                obj=QD(eye(m,n),zeros(m,n),zeros(m,n),zeros(m,n));
            end
        end
        
        function obj = qdpi()
            obj=QD(3.141592653589793116,1.224646799147353207e-16,-2.994769809718339666e-33,1.112454220863365282e-49);
        end
        
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
        
        
        %----------------%
        %    operetor    %
        %----------------%
        %a+b
        function obj = plus(a, b)
            %             global deffma;
            global defavx;
            global defthreadNum;
            %             fma = deffma;
            avx = defavx;
            threadNum = defthreadNum;
            obj = QD(0);
            [m,n]=size(a);
            isN1 = isnumeric(a);
            isN2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            isQD1 = isa(a, 'QD');
            isQD2 = isa(b, 'QD');
            
            if(isN1)
                if(isQD2)%double+QD
                    [obj.hh,obj.hl,obj.lh,obj.ll] = d_a_qd(a,b.hh,b.hl,b.lh,b.ll,threadNum,avx);
                else
                    %obj2 is not supported
                end
            elseif(isDD1)%DD+
                if(isQD2)%QD
                    [obj.hh,obj.hl,obj.lh,obj.ll] = dd_a_qd(a.hi,a.lo,b.hh,b.hl,b.lh,b.ll,threadNum,avx);
                end
            elseif(isQD1)
                if(isN2)%QD+double
                    [obj.hh,obj.hl,obj.lh,obj.ll] = d_a_qd(b,a.hh,a.hl,a.lh,a.ll,threadNum,avx);
                elseif(isDD2)%QD+DD
                    [obj.hh,obj.hl,obj.lh,obj.ll] = dd_a_qd(b.hi,b.lo,a.hh,a.hl,a.lh,a.ll,threadNum,avx);
                elseif(isQD2)%QD+QD
                    if (m==1&&n==1)
                        
                        [s1, t1] = Calc.twoSum(a.hh, b.hh);
                        [s2, t2] = Calc.twoSum(a.hl, b.hl);
                        [s3, t3] = Calc.twoSum(a.lh, b.lh);
                        [s4, t4] = Calc.twoSum(a.ll, b.ll);
                        [s2, t1] = Calc.twoSum(s2,t1);
                        [s3,t1,t2] = Calc.threeSum(s3,t2,t1);
                        [s4,t1] = Calc.threeSum2(s4,t3,t1);
                        s5=t1+t2+t4;
                        [obj.hh,obj.hl,obj.lh,obj.ll]=Calc.renormalize(s1,s2,s3,s4,s5);
                    else
                        [obj.hh,obj.hl,obj.lh,obj.ll] = qd_a_qd(a.hh,a.hl,a.lh,a.ll,b.hh,b.hl,b.lh,b.ll,threadNum,avx);
                    end
                else
                    %obj2 is not supported
                end
            else
                %obj1 is not supported
            end
        end
        %a-b
        function obj = minus(a, b)
            obj=plus(a,-b);
        end
        %-a
        function obj = uminus(a)
            obj = QD(0);
            obj.hh = -a.hh;
            obj.hl = -a.hl;
            obj.lh = -a.lh;
            obj.ll = -a.ll;
        end
        %+a
        function obj = uplus(a)
            obj = QD(0);
            obj.hh = a.hh;
            obj.hl = a.hl;
            obj.lh = a.lh;
            obj.ll = a.ll;
        end
        %a*b
        function obj=mtimes(a,b)
            global deffma;
            global defavx;
            global defthreadNum;
            fma = deffma;
            avx = defavx;
            threadNum = defthreadNum;
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            
            if(n1 ~= m2)
                if((m1==1&&n1==1)||(m2==1&&n2==1))||(n1==1&&n2==1)
                    
                else
                    %unsupported operation
                    disp('Matrix dimension must agree!');
                end
            end
            obj = QD(zeros(m1,n2));
            
            isN1 = isnumeric(a);
            isN2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            isQD1 = isa(a, 'QD');
            isQD2 = isa(b, 'QD');
            
            
            if(isN1)
                if(isQD2)%double*QD
                    if (m1 == 1 && n1 == 1) && (m2 == 1 && n2 == 1)
                        [p1,e1] = Calc.twoProd(a,b.hh);
                        [p2,e2] = Calc.twoProd(a,b.hl);
                        [p3,e3] = Calc.twoProd(a,b.lh);
                        p4 = a*b.ll;
                        [p2,e1] = Calc.twoSum(p2,e1);
                        [p3,e2,e1] = Calc.threeSum(p3,e2,e1);
                        [p4,e2] = Calc.threeSum2(p4,e3,e2);
                        p5 = e1+e2;
                        [obj.hh,obj.hl,obj.lh,obj.ll] = Calc.renormalize(p1,p2,p3,p4,p5);
                    elseif(m1 == 1 && n1 == 1)
                        obj = QD(zeros(m2, n2));
                        if (m2 == 1)
                            b = b';
                            [obj.hh,obj.hl,obj.lh,obj.ll]=d_scl_qd(a,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                            obj = obj';
                        else
                            [obj.hh,obj.hl,obj.lh,obj.ll]=d_scl_qd(a,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                        end
                    elseif(m2 == 1 && n2 == 1)
                        obj = QD(zeros(m1, n1));
                        if (m1 == 1)
                            b = b';
                            [obj.hh,obj.hl,obj.lh,obj.ll]=d_scl_qd(a,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                            obj = obj';
                        else
                            [obj.hh,obj.hl,obj.lh,obj.ll]=d_scl_qd(a,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                        end
                    elseif(m1 == 1 && n2 == 1)
                        a=a';
                        obj = QD(zeros(m1,n2));
                        [obj.hh,obj.hl,obj.lh,obj.ll]=d_dot_qd(a,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                    elseif n1 == m2 && n2 == 1
                        [obj.hh,obj.hl,obj.lh,obj.ll]=d_mv_qd(a,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                    elseif m1 == 1 && n2 == m2
                        a = a';
                        obj = QD(zeros(m1,n2));
                        [obj.hh,obj.hl,obj.lh,obj.ll]=qd_tmv_d(b.hh,b.hl,b.lh,b.ll,a,threadNum,avx,fma);
                        obj = obj';
                    elseif n1 == m2
                        [obj.hh,obj.hl,obj.lh,obj.ll]=d_mm_qd(a,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                    else
                        
                    end
                else
                    %obj2 is not supported
                end
            elseif(isDD1)
                if(isQD2)%DD*QD
                    if (m1 == 1 && n1 == 1) && (m2 == 1 && n2 == 1)
                        [p1,e1] = Calc.twoProd(a.hi,b.hh);
                        [p2,e2] = Calc.twoProd(a.lo,b.hh);
                        [p3,e3] = Calc.twoProd(a.hi,b.hl);
                        [p4,e4] = Calc.twoProd(a.lo,b.hl);
                        [p5,e5] = Calc.twoProd(a.hi,b.lh);
                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                        [p3,p4,p5] = Calc.threeSum(p3,p4,p5);
                        [e2,e3] = Calc.twoSum(e2,e3);
                        [p3,e2] = Calc.twoSum(p3,e2);
                        [p4,e3] = Calc.twoSum(p4,e3);
                        [p4,e2] = Calc.twoSum(p4,e2);
                        p5 = p5+e2+e3;
                        s = a.lo*b.lh + a.hi*b.ll + e4 + e5;
                        [p4,e1] = Calc.threeSum2(p4,e1,s);
                        p5 = p5+e1;
                        [obj.hh,obj.hl,obj.lh,obj.ll] = Calc.renormalize(p1,p2,p3,p4,p5);
                        
                    elseif(m1 == 1 && n1 == 1)
                        obj = QD(zeros(m2, n2));
                        if (m2 == 1)
                            b = b';
                            [obj.hh,obj.hl,obj.lh,obj.ll]=dd_scl_qd(a.hi,a.lo,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                            obj = obj';
                        else
                            [obj.hh,obj.hl,obj.lh,obj.ll]=dd_scl_qd(a.hi,a.lo,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                        end
                    elseif(m2 == 1 && n2 == 1)
                        obj = QD(zeros(m1, n1));
                        if (m1 == 1)
                            b = b';
                            [obj.hh,obj.hl,obj.lh,obj.ll]=dd_scl_qd(a.hi,a.lo,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                            obj = obj';
                        else
                            [obj.hh,obj.hl,obj.lh,obj.ll]=dd_scl_qd(a.hi,a.lo,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                        end
                    elseif(m1 == 1 && n2 == 1)
                        a=a';
                        obj = QD(zeros(m1,n2));
                        [obj.hh,obj.hl,obj.lh,obj.ll]=dd_dot_qd(a.hi,a.lo,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                    elseif n1 == m2 && n2 == 1
                        [obj.hh,obj.hl,obj.lh,obj.ll]=dd_mv_qd(a.hi,a.lo,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                    elseif m1 == 1 && n2 == m2
                        a = a';
                        obj = QD(zeros(m1,n2));
                        [obj.hh,obj.hl,obj.lh,obj.ll]=qd_tmv_dd(b.hh,b.hl,b.lh,b.ll,a.hi,a.lo,threadNum,avx,fma);
                        obj = obj';
                    elseif n1 == m2
                        [obj.hh,obj.hl,obj.lh,obj.ll]=dd_mm_qd(a.hi,a.lo,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                    else
                        
                    end
                end
            elseif(isQD1)
                if(isN2)%QD*double
                    if (m1 == 1 && n1 == 1) && (m2 == 1 && n2 == 1)
                        [p1,e1] = Calc.twoProd(a.hh,b);
                        [p2,e2] = Calc.twoProd(a.hl,b);
                        [p3,e3] = Calc.twoProd(a.lh,b);
                        p4 = a.ll*b;
                        [p2,e1] = Calc.twoSum(p2,e1);
                        [p3,e2,e1] = Calc.threeSum(p3,e2,e1);
                        [p4,e2] = Calc.threeSum2(p4,e3,e2);
                        p5 = e1+e2;
                        [obj.hh,obj.hl,obj.lh,obj.ll] = Calc.renormalize(p1,p2,p3,p4,p5);
                    elseif(m1 == 1 && n1 == 1)
                        obj = QD(zeros(m2, n2));
                        if (n2 == 1)
                            [obj.hh,obj.hl,obj.lh,obj.ll]=qd_scl_d(a.hh,a.hl,a.lh,a.ll,b,threadNum,avx,fma);
                        else
                            b = b';
                            [obj.hh,obj.hl,obj.lh,obj.ll]=qd_scl_d(a.hh,a.hl,a.lh,a.ll,b,threadNum,avx,fma);
                            obj = obj';
                        end
                    elseif(m2 == 1 && n2 == 1)
                        obj = QD(zeros(m1, n1));
                        if (m1 == 1)
                            b = b';
                            [obj.hh,obj.hl,obj.lh,obj.ll]=qd_scl_d(a.hh,a.hl,a.lh,a.ll,b,threadNum,avx,fma);
                            obj = obj';
                        else
                            [obj.hh,obj.hl,obj.lh,obj.ll]=qd_scl_d(a.hh,a.hl,a.lh,a.ll,b,threadNum,avx,fma);
                        end
                    elseif(m1 == 1 && n2 == 1)
                        a=a';
                        obj = QD(zeros(m1,n2));
                        [obj.hh,obj.hl,obj.lh,obj.ll]=d_dot_qd(b,a.hh,a.hl,a.lh,a.ll,threadNum,avx,fma);
                    elseif n1 == m2 && n2 == 1
                        [obj.hh,obj.hl,obj.lh,obj.ll]=qd_mv_d(a.hh,a.hl,a.lh,a.ll,b,threadNum,avx,fma);
                    elseif m1 == 1 && n2 == m2
                        a = a';
                        obj = QD(zeros(m1,n2));
                        [obj.hh,obj.hl,obj.lh,obj.ll]=d_tmv_qd(b,a.hh,a.hl,a.lh,a.ll,threadNum,avx,fma);
                        obj = obj';
                    elseif n1 == m2
                        [obj.hh,obj.hl,obj.lh,obj.ll]=qd_mm_d(a.hh,a.hl,a.lh,a.ll,b,threadNum,avx,fma);
                    else
                        
                    end
                elseif(isDD2)%QD*DD
                    if (m1 == 1 && n1 == 1) && (m2 == 1 && n2 == 1)
                        [p1,e1] = Calc.twoProd(a.hh,b.hi);
                        [p2,e2] = Calc.twoProd(a.hh,b.lo);
                        [p3,e3] = Calc.twoProd(a.hl,b.hi);
                        [p4,e4] = Calc.twoProd(a.hl,b.lo);
                        [p5,e5] = Calc.twoProd(a.lh,b.hi);
                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                        [p3,p4,p5] = Calc.threeSum(p3,p4,p5);
                        [e2,e3] = Calc.twoSum(e2,e3);
                        [p3,e2] = Calc.twoSum(p3,e2);
                        [p4,e3] = Calc.twoSum(p4,e3);
                        [p4,e2] = Calc.twoSum(p4,e2);
                        p5 = p5+e2+e3;
                        s = a.lh*b.lo + a.ll*b.hi + e4 + e5;
                        [p4,e1] = Calc.threeSum2(p4,e1,s);
                        p5 = p5+e1;
                        [obj.hh,obj.hl,obj.lh,obj.ll] = Calc.renormalize(p1,p2,p3,p4,p5);
                    elseif ((m1 == 1 && n1 == 1))
                        obj = QD(zeros(m2, n2));
                        if (m2 == 1)
                            b = b';
                            [obj.hh,obj.hl,obj.lh,obj.ll]=qd_scl_dd(a.hh,a.hl,a.lh,a.ll,b.hi,b.lo,threadNum,avx,fma);
                            obj = obj';
                        else
                            [obj.hh,obj.hl,obj.lh,obj.ll]=qd_scl_dd(a.hh,a.hl,a.lh,a.ll,b.hi,b.lo,threadNum,avx,fma);
                        end
                    elseif ((m2 == 1 && n2 == 1))
                        obj = QD(zeros(m1, n1));
                        if (m1 == 1)
                            b = b';
                            [obj.hh,obj.hl,obj.lh,obj.ll]=qd_scl_dd(a.hh,a.hl,a.lh,a.ll,b.hi,b.lo,threadNum,avx,fma);
                            obj = obj';
                        else
                            [obj.hh,obj.hl,obj.lh,obj.ll]=qd_scl_dd(a.hh,a.hl,a.lh,a.ll,b.hi,b.lo,threadNum,avx,fma);
                        end
                    elseif(m1 == 1 && n2 == 1)
                        a=a';
                        obj = QD(zeros(m1,n2));
                        [obj.hh,obj.hl,obj.lh,obj.ll]=dd_dot_qd(b.hi,b.lo,a.hh,a.hl,a.lh,a.ll,threadNum,avx,fma);
                    elseif n1 == m2 && n2 == 1
                        [obj.hh,obj.hl,obj.lh,obj.ll]=qd_mv_dd(a.hh,a.hl,a.lh,a.ll,b.hi,b.lo,threadNum,avx,fma);
                    elseif m1 == 1 && n2 == m2
                        a=a';
                        obj = QD(zeros(n1,m1));
                        [obj.hh,obj.hl,obj.lh,obj.ll]=dd_tmv_qd(b.hi,b.lo,a.hh,a.hl,a.lh,a.ll,threadNum,avx,fma);
                        obj = obj';
                    elseif n1 == m2
                        [obj.hh,obj.hl,obj.lh,obj.ll]=qd_mm_dd(a.hh,a.hl,a.lh,a.ll,b.hi,b.lo,threadNum,avx,fma);
                    else
                        
                    end
                elseif(isQD2)%QD*QD
                    %if scalar
                    if((m1==1&&n1==1)&&(m2==1&&n2==1))
                        [p1,e1] = Calc.twoProd(a.hh,b.hh);
                        [p2,e2] = Calc.twoProd(a.hh,b.hl);
                        [p3,e3] = Calc.twoProd(a.hl,b.hh);
                        [p4,e4] = Calc.twoProd(a.hh,b.lh);
                        [p5,e5] = Calc.twoProd(a.hl,b.hl);
                        [p6,e6] = Calc.twoProd(a.lh,b.hh);
                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                        
                        [p4,p5,p6] = Calc.threeSum(p4,p5,p6);
                        [p3,e2,e3] = Calc.threeSum(p3,e2,e3);
                        [p3,p4] = Calc.twoSum(p3,p4);
                        [p5,e2] = Calc.twoSum(e2,p5);
                        [p4,p5] = Calc.twoSum(p4,p5);
                        p5 = p5+e2+e3+p6;
                        p4 = p4+a.hh*b.ll+a.hl*b.lh+a.lh*b.hl+a.ll*b.hh+e1+e4+e5+e6;
                        [obj.hh,obj.hl,obj.lh,obj.ll] = Calc.renormalize(p1,p2,p3,p4,p5);
                    elseif(m1==1&&n1==1)
                        if(m2==1)
                            b = b';
                            [obj.hh,obj.hl,obj.lh,obj.ll]=qd_scl_qd(a.hh,a.hl,a.lh,a.ll,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                            obj = obj';
                        else
                            [obj.hh,obj.hl,obj.lh,obj.ll]=qd_scl_qd(a.hh,a.hl,a.lh,a.ll,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                        end
                    elseif(m2==1&&n2==1)
                        if(m1==1)
                            b = b';
                            [obj.hh,obj.hl,obj.lh,obj.ll]=qd_scl_qd(a.hh,a.hl,a.lh,a.ll,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                            obj = obj';
                        else
                            [obj.hh,obj.hl,obj.lh,obj.ll]=qd_scl_qd(a.hh,a.hl,a.lh,a.ll,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                        end
                    elseif(m1 == 1 && n2 == 1)
                        a=a';
                        obj = QD(zeros(m1,n2));
                        [obj.hh,obj.hl,obj.lh,obj.ll]=qd_dot_qd(a.hh,a.hl,a.lh,a.ll,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                    elseif (n1==m2&&n2==1) %(m,n)*(n,1)
                        
                        [obj.hh,obj.hl,obj.lh,obj.ll]=qd_mv_qd(a.hh,a.hl,a.lh,a.ll,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                        
                    elseif(m1==1&&n1==m2) %(1,l)*(l,n)
                        a=a';
                        obj = QD(zeros(n1, m1));
                        [obj.hh,obj.hl,obj.lh,obj.ll]=qd_tmv_qd(b.hh,b.hl,b.lh,b.ll,a.hh,a.hl,a.lh,a.ll,threadNum,avx,fma);
                        obj=obj';
                    elseif (n1==m2)%(m1,n1)*(m2,n2)
                        [obj.hh,obj.hl,obj.lh,obj.ll]=qd_mm_qd(a.hh,a.hl,a.lh,a.ll,b.hh,b.hl,b.lh,b.ll,threadNum,avx,fma);
                    else
                        %obj2 is not supported
                    end
                else
                    %obj1 is not supported
                end
            end
        end
        %a.*b
        function obj = times(a, b)
            [na,ma]=size(a);
            [nb,mb]=size(b);
            is1=na~=1;
            is2=ma~=1;
            is3=nb~=1;
            is4=mb~=1;
            isaDD=isa(a,'DD');
            isaQD=isa(a,'QD');
            isbDD=isa(b,'DD');
            isbQD=isa(b,'QD');
            isaD=isnumeric(a);
            isbD=isnumeric(b);
            if isaD&&isbD
            elseif isaD&&~isbD
                warning('off','MATLAB:structOnObject');
                b=struct(b);
            elseif ~isaD&&isbD
                warning('off','MATLAB:structOnObject');
                a=struct(a);
            else
                warning('off','MATLAB:structOnObject');
                a=struct(a);
                b=struct(b);
            end
            if (is1&&is2&&is3&&is4)%not vector
                if (nb~=na)||(ma~=mb)
                    disp('Matrix dimensions must agree');
                    obj=0;
                else
                    obj=QD(zeros(na,mb));
                    if isaQD&&isbQD
                        for i=1:na
                            for j=1:mb
                                [p1,e1] = Calc.twoProd(a.hh(i,j),b.hh(i,j));
                                [p2,e2] = Calc.twoProd(a.hh(i,j),b.hl(i,j));
                                [p3,e3] = Calc.twoProd(a.hl(i,j),b.hh(i,j));
                                [p4,e4] = Calc.twoProd(a.hh(i,j),b.lh(i,j));
                                [p5,e5] = Calc.twoProd(a.hl(i,j),b.hl(i,j));
                                [p6,e6] = Calc.twoProd(a.lh(i,j),b.hh(i,j));
                                [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                [p4,p5,p6] = Calc.threeSum(p4,p5,p6);
                                [p3,e2,e3] = Calc.threeSum(p3,e2,e3);
                                [p3,p4] = Calc.twoSum(p3,p4);
                                [p5,e2] = Calc.twoSum(e2,p5);
                                [p4,p5] = Calc.twoSum(p4,p5);
                                p5 = p5+e2+e3+p6;
                                p4 = p4+a.hh(i,j)*b.ll(i,j)+a.hl(i,j)*b.lh(i,j)+a.lh(i,j)*b.hl(i,j)+a.ll(i,j)*b.hh(i,j)+e1+e4+e5+e6;
                                [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                            end
                        end
                    elseif isaQD&&isbDD
                        for i=1:na
                            for j=1:mb
                                [p1,e1] = Calc.twoProd(a.hh(i,j),b.hi(i,j));
                                [p2,e2] = Calc.twoProd(a.hh(i,j),b.lo(i,j));
                                [p3,e3] = Calc.twoProd(a.hl(i,j),b.hi(i,j));
                                [p4,e4] = Calc.twoProd(a.hl(i,j),b.lo(i,j));
                                [p5,e5] = Calc.twoProd(a.lh(i,j),b.hi(i,j));
                                [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                [p3,p4,p5] = Calc.threeSum(p3,p4,p5);
                                [e2,e3] = Calc.twoSum(e2,e3);
                                [p3,e2] = Calc.twoSum(p3,e2);
                                [p4,e3] = Calc.twoSum(p4,e3);
                                [p4,e2] = Calc.twoSum(p4,e2);
                                p5 = p5+e2+e3;
                                s = a.lh(i,j)*b.lo(i,j) + a.ll(i,j)*b.hi(i,j) + e4 + e5;
                                [p4,e1] = Calc.threeSum2(p4,e1,s);
                                p5 = p5+e1;
                                [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                            end
                        end
                    elseif isaDD&&isbQD
                        for i=1:na
                            for j=1:mb
                                [p1,e1] = Calc.twoProd(b.hh(i,j),a.hi(i,j));
                                [p2,e2] = Calc.twoProd(b.hh(i,j),a.lo(i,j));
                                [p3,e3] = Calc.twoProd(b.hl(i,j),a.hi(i,j));
                                [p4,e4] = Calc.twoProd(b.hl(i,j),a.lo(i,j));
                                [p5,e5] = Calc.twoProd(b.lh(i,j),a.hi(i,j));
                                [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                [p3,p4,p5] = Calc.threeSum(p3,p4,p5);
                                [e2,e3] = Calc.twoSum(e2,e3);
                                [p3,e2] = Calc.twoSum(p3,e2);
                                [p4,e3] = Calc.twoSum(p4,e3);
                                [p4,e2] = Calc.twoSum(p4,e2);
                                p5 = p5+e2+e3;
                                s = b.lh(i,j)*a.lo(i,j) + b.ll(i,j)*a.hi(i,j) + e4 + e5;
                                [p4,e1] = Calc.threeSum2(p4,e1,s);
                                p5 = p5+e1;
                                [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                            end
                        end
                    elseif isaQD&&isbD
                        for i=1:na
                            for j=1:mb
                                [p1,e1] = Calc.twoProd(a.hh(i,j),b(i,j));
                                [p2,e2] = Calc.twoProd(a.hl(i,j),b(i,j));
                                [p3,e3] = Calc.twoProd(a.lh(i,j),b(i,j));
                                p4 = a.ll(i,j)*b(i,j);
                                [p2,e1] = Calc.twoSum(p2,e1);
                                [p3,e2,e1] = Calc.threeSum(p3,e2,e1);
                                [p4,e2] = Calc.threeSum2(p4,e3,e2);
                                p5 = e1+e2;
                                [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                            end
                        end
                    elseif isaD&&isbQD
                        for i=1:na
                            for j=1:mb
                                [p1,e1] = Calc.twoProd(b.hh(i,j),a(i,j));
                                [p2,e2] = Calc.twoProd(b.hl(i,j),a(i,j));
                                [p3,e3] = Calc.twoProd(b.lh(i,j),a(i,j));
                                p4 = b.ll(i,j)*a(i,j);
                                [p2,e1] = Calc.twoSum(p2,e1);
                                [p3,e2,e1] = Calc.threeSum(p3,e2,e1);
                                [p4,e2] = Calc.threeSum2(p4,e3,e2);
                                p5 = e1+e2;
                                [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                            end
                        end
                    else
                        
                    end
                end
            else %include vector
                if (is1)&&(is2)
                    if is3 %mb==1
                        if nb==na||nb==ma
                            obj=QD(zeros(na,ma));
                            if isaQD&&isbQD
                                for i=1:na
                                    for j=1:ma
                                        [p1,e1] = Calc.twoProd(a.hh(i,j),b.hh(i,1));
                                        [p2,e2] = Calc.twoProd(a.hh(i,j),b.hl(i,1));
                                        [p3,e3] = Calc.twoProd(a.hl(i,j),b.hh(i,1));
                                        [p4,e4] = Calc.twoProd(a.hh(i,j),b.lh(i,1));
                                        [p5,e5] = Calc.twoProd(a.hl(i,j),b.hl(i,1));
                                        [p6,e6] = Calc.twoProd(a.lh(i,j),b.hh(i,1));
                                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                        [p4,p5,p6] = Calc.threeSum(p4,p5,p6);
                                        [p3,e2,e3] = Calc.threeSum(p3,e2,e3);
                                        [p3,p4] = Calc.twoSum(p3,p4);
                                        [p5,e2] = Calc.twoSum(e2,p5);
                                        [p4,p5] = Calc.twoSum(p4,p5);
                                        p5 = p5+e2+e3+p6;
                                        p4 = p4+a.hh(i,j)*b.ll(i,1)+a.hl(i,j)*b.lh(i,1)+a.lh(i,j)*b.hl(i,1)+a.ll(i,j)*b.hh(i,1)+e1+e4+e5+e6;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaQD&&isbDD
                                for i=1:na
                                    for j=1:ma
                                        [p1,e1] = Calc.twoProd(a.hh(i,j),b.hi(i,1));
                                        [p2,e2] = Calc.twoProd(a.hh(i,j),b.lo(i,1));
                                        [p3,e3] = Calc.twoProd(a.hl(i,j),b.hi(i,1));
                                        [p4,e4] = Calc.twoProd(a.hl(i,j),b.lo(i,1));
                                        [p5,e5] = Calc.twoProd(a.lh(i,j),b.hi(i,1));
                                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                        [p3,p4,p5] = Calc.threeSum(p3,p4,p5);
                                        [e2,e3] = Calc.twoSum(e2,e3);
                                        [p3,e2] = Calc.twoSum(p3,e2);
                                        [p4,e3] = Calc.twoSum(p4,e3);
                                        [p4,e2] = Calc.twoSum(p4,e2);
                                        p5 = p5+e2+e3;
                                        s = a.lh(i,j)*b.lo(i,1) + a.ll(i,j)*b.hi(i,1) + e4 + e5;
                                        [p4,e1] = Calc.threeSum2(p4,e1,s);
                                        p5 = p5+e1;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaDD&&isbQD
                                for i=1:na
                                    for j=1:ma
                                        [p1,e1] = Calc.twoProd(b.hh(i,1),a.hi(i,j));
                                        [p2,e2] = Calc.twoProd(b.hh(i,1),a.lo(i,j));
                                        [p3,e3] = Calc.twoProd(b.hl(i,1),a.hi(i,j));
                                        [p4,e4] = Calc.twoProd(b.hl(i,1),a.lo(i,j));
                                        [p5,e5] = Calc.twoProd(b.lh(i,1),a.hi(i,j));
                                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                        [p3,p4,p5] = Calc.threeSum(p3,p4,p5);
                                        [e2,e3] = Calc.twoSum(e2,e3);
                                        [p3,e2] = Calc.twoSum(p3,e2);
                                        [p4,e3] = Calc.twoSum(p4,e3);
                                        [p4,e2] = Calc.twoSum(p4,e2);
                                        p5 = p5+e2+e3;
                                        s = b.lh(i,1)*a.lo(i,j) + b.ll(i,1)*a.hi(i,j) + e4 + e5;
                                        [p4,e1] = Calc.threeSum2(p4,e1,s);
                                        p5 = p5+e1;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaQD&&isbD
                                for i=1:na
                                    for j=1:ma
                                        [p1,e1] = Calc.twoProd(a.hh(i,j),b(i,1));
                                        [p2,e2] = Calc.twoProd(a.hl(i,j),b(i,1));
                                        [p3,e3] = Calc.twoProd(a.lh(i,j),b(i,1));
                                        p4 = a.ll(i,j)*b(i,1);
                                        [p2,e1] = Calc.twoSum(p2,e1);
                                        [p3,e2,e1] = Calc.threeSum(p3,e2,e1);
                                        [p4,e2] = Calc.threeSum2(p4,e3,e2);
                                        p5 = e1+e2;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaD&&isbQD
                                for i=1:na
                                    for j=1:ma
                                        [p1,e1] = Calc.twoProd(b.hh(i,1),a(i,j));
                                        [p2,e2] = Calc.twoProd(b.hl(i,1),a(i,j));
                                        [p3,e3] = Calc.twoProd(b.lh(i,1),a(i,j));
                                        p4 = b.ll(i,1)*a(i,j);
                                        [p2,e1] = Calc.twoSum(p2,e1);
                                        [p3,e2,e1] = Calc.threeSum(p3,e2,e1);
                                        [p4,e2] = Calc.threeSum2(p4,e3,e2);
                                        p5 = e1+e2;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            else
                                
                            end
                        else
                            disp('Matrix dimensions must agree');
                            obj=0;
                        end
                    elseif is4 %nb==1
                        if mb==na||mb==ma
                            obj=QD(zeros(na,ma));
                            if isaQD&&isbQD
                                for i=1:na
                                    for j=1:ma
                                        [p1,e1] = Calc.twoProd(a.hh(i,j),b.hh(1,j));
                                        [p2,e2] = Calc.twoProd(a.hh(i,j),b.hl(1,j));
                                        [p3,e3] = Calc.twoProd(a.hl(i,j),b.hh(1,j));
                                        [p4,e4] = Calc.twoProd(a.hh(i,j),b.lh(1,j));
                                        [p5,e5] = Calc.twoProd(a.hl(i,j),b.hl(1,j));
                                        [p6,e6] = Calc.twoProd(a.lh(i,j),b.hh(1,j));
                                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                        [p4,p5,p6] = Calc.threeSum(p4,p5,p6);
                                        [p3,e2,e3] = Calc.threeSum(p3,e2,e3);
                                        [p3,p4] = Calc.twoSum(p3,p4);
                                        [p5,e2] = Calc.twoSum(e2,p5);
                                        [p4,p5] = Calc.twoSum(p4,p5);
                                        p5 = p5+e2+e3+p6;
                                        p4 = p4+a.hh(i,j)*b.ll(1,j)+a.hl(i,j)*b.lh(1,j)+a.lh(i,j)*b.hl(1,j)+a.ll(i,j)*b.hh(1,j)+e1+e4+e5+e6;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                                
                            elseif isaQD&&isbDD
                                for i=1:na
                                    for j=1:ma
                                        [p1,e1] = Calc.twoProd(a.hh(i,j),b.hi(1,j));
                                        [p2,e2] = Calc.twoProd(a.hh(i,j),b.lo(1,j));
                                        [p3,e3] = Calc.twoProd(a.hl(i,j),b.hi(1,j));
                                        [p4,e4] = Calc.twoProd(a.hl(i,j),b.lo(1,j));
                                        [p5,e5] = Calc.twoProd(a.lh(i,j),b.hi(1,j));
                                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                        [p3,p4,p5] = Calc.threeSum(p3,p4,p5);
                                        [e2,e3] = Calc.twoSum(e2,e3);
                                        [p3,e2] = Calc.twoSum(p3,e2);
                                        [p4,e3] = Calc.twoSum(p4,e3);
                                        [p4,e2] = Calc.twoSum(p4,e2);
                                        p5 = p5+e2+e3;
                                        s = a.lh(i,j)*b.lo(1,j) + a.ll(i,j)*b.hi(1,j) + e4 + e5;
                                        [p4,e1] = Calc.threeSum2(p4,e1,s);
                                        p5 = p5+e1;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaDD&&isbQD
                                for i=1:na
                                    for j=1:ma
                                        [p1,e1] = Calc.twoProd(b.hh(1,j),a.hi(i,j));
                                        [p2,e2] = Calc.twoProd(b.hh(1,j),a.lo(i,j));
                                        [p3,e3] = Calc.twoProd(b.hl(1,j),a.hi(i,j));
                                        [p4,e4] = Calc.twoProd(b.hl(1,j),a.lo(i,j));
                                        [p5,e5] = Calc.twoProd(b.lh(1,j),a.hi(i,j));
                                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                        [p3,p4,p5] = Calc.threeSum(p3,p4,p5);
                                        [e2,e3] = Calc.twoSum(e2,e3);
                                        [p3,e2] = Calc.twoSum(p3,e2);
                                        [p4,e3] = Calc.twoSum(p4,e3);
                                        [p4,e2] = Calc.twoSum(p4,e2);
                                        p5 = p5+e2+e3;
                                        s = b.lh(1,j)*a.lo(i,j) + b.ll(1,j)*a.hi(i,j) + e4 + e5;
                                        [p4,e1] = Calc.threeSum2(p4,e1,s);
                                        p5 = p5+e1;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaQD&&isbD
                                for i=1:na
                                    for j=1:ma
                                        [p1,e1] = Calc.twoProd(a.hh(i,j),b(1,j));
                                        [p2,e2] = Calc.twoProd(a.hl(i,j),b(1,j));
                                        [p3,e3] = Calc.twoProd(a.lh(i,j),b(1,j));
                                        p4 = a.ll(i,j)*b(1,j);
                                        [p2,e1] = Calc.twoSum(p2,e1);
                                        [p3,e2,e1] = Calc.threeSum(p3,e2,e1);
                                        [p4,e2] = Calc.threeSum2(p4,e3,e2);
                                        p5 = e1+e2;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaD&&isbQD
                                for i=1:na
                                    for j=1:ma
                                        [p1,e1] = Calc.twoProd(b.hh(1,j),a(i,j));
                                        [p2,e2] = Calc.twoProd(b.hl(1,j),a(i,j));
                                        [p3,e3] = Calc.twoProd(b.lh(1,j),a(i,j));
                                        p4 = b.ll(a,j)*a(i,j);
                                        [p2,e1] = Calc.twoSum(p2,e1);
                                        [p3,e2,e1] = Calc.threeSum(p3,e2,e1);
                                        [p4,e2] = Calc.threeSum2(p4,e3,e2);
                                        p5 = e1+e2;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            else
                                
                            end
                        else
                            disp('Matrix dimensions must agree');
                            obj=0;
                        end
                    end
                else
                    if is1 %ma==1
                        if nb==na||nb==ma
                            obj=QD(zeros(nb,mb));
                            if isaQD&&isbQD
                                for i=1:nb
                                    for j=1:mb
                                        [p1,e1] = Calc.twoProd(a.hh(i,1),b.hh(i,j));
                                        [p2,e2] = Calc.twoProd(a.hh(i,1),b.hl(i,j));
                                        [p3,e3] = Calc.twoProd(a.hl(i,1),b.hh(i,j));
                                        [p4,e4] = Calc.twoProd(a.hh(i,1),b.lh(i,j));
                                        [p5,e5] = Calc.twoProd(a.hl(i,1),b.hl(i,j));
                                        [p6,e6] = Calc.twoProd(a.lh(i,1),b.hh(i,j));
                                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                        [p4,p5,p6] = Calc.threeSum(p4,p5,p6);
                                        [p3,e2,e3] = Calc.threeSum(p3,e2,e3);
                                        [p3,p4] = Calc.twoSum(p3,p4);
                                        [p5,e2] = Calc.twoSum(e2,p5);
                                        [p4,p5] = Calc.twoSum(p4,p5);
                                        p5 = p5+e2+e3+p6;
                                        p4 = p4+a.hh(i,1)*b.ll(i,j)+a.hl(i,1)*b.lh(i,j)+a.lh(i,1)*b.hl(i,j)+a.ll(i,1)*b.hh(i,j)+e1+e4+e5+e6;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaQD&&isbDD
                                for i=1:nb
                                    for j=1:mb
                                        [p1,e1] = Calc.twoProd(a.hh(i,1),b.hi(i,j));
                                        [p2,e2] = Calc.twoProd(a.hh(i,1),b.lo(i,j));
                                        [p3,e3] = Calc.twoProd(a.hl(i,1),b.hi(i,j));
                                        [p4,e4] = Calc.twoProd(a.hl(i,1),b.lo(i,j));
                                        [p5,e5] = Calc.twoProd(a.lh(i,1),b.hi(i,j));
                                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                        [p3,p4,p5] = Calc.threeSum(p3,p4,p5);
                                        [e2,e3] = Calc.twoSum(e2,e3);
                                        [p3,e2] = Calc.twoSum(p3,e2);
                                        [p4,e3] = Calc.twoSum(p4,e3);
                                        [p4,e2] = Calc.twoSum(p4,e2);
                                        p5 = p5+e2+e3;
                                        s = a.lh(i,1)*b.lo(i,j) + a.ll(i,1)*b.hi(i,j) + e4 + e5;
                                        [p4,e1] = Calc.threeSum2(p4,e1,s);
                                        p5 = p5+e1;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaDD&&isbQD
                                for i=1:nb
                                    for j=1:mb
                                        [p1,e1] = Calc.twoProd(b.hh(i,j),a.hi(i,1));
                                        [p2,e2] = Calc.twoProd(b.hh(i,j),a.lo(i,1));
                                        [p3,e3] = Calc.twoProd(b.hl(i,j),a.hi(i,1));
                                        [p4,e4] = Calc.twoProd(b.hl(i,j),a.lo(i,1));
                                        [p5,e5] = Calc.twoProd(b.lh(i,j),a.hi(i,1));
                                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                        [p3,p4,p5] = Calc.threeSum(p3,p4,p5);
                                        [e2,e3] = Calc.twoSum(e2,e3);
                                        [p3,e2] = Calc.twoSum(p3,e2);
                                        [p4,e3] = Calc.twoSum(p4,e3);
                                        [p4,e2] = Calc.twoSum(p4,e2);
                                        p5 = p5+e2+e3;
                                        s = b.lh(i,j)*a.lo(i,1) + b.ll(i,j)*a.hi(i,1) + e4 + e5;
                                        [p4,e1] = Calc.threeSum2(p4,e1,s);
                                        p5 = p5+e1;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaQD&&isbD
                                for i=1:nb
                                    for j=1:mb
                                        [p1,e1] = Calc.twoProd(a.hh(i,1),b(i,j));
                                        [p2,e2] = Calc.twoProd(a.hl(i,1),b(i,j));
                                        [p3,e3] = Calc.twoProd(a.lh(i,1),b(i,j));
                                        p4 = a.ll(i,1)*b(i,j);
                                        [p2,e1] = Calc.twoSum(p2,e1);
                                        [p3,e2,e1] = Calc.threeSum(p3,e2,e1);
                                        [p4,e2] = Calc.threeSum2(p4,e3,e2);
                                        p5 = e1+e2;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaD&&isbQD
                                for i=1:nb
                                    for j=1:mb
                                        [p1,e1] = Calc.twoProd(b.hh(i,j),a(i,1));
                                        [p2,e2] = Calc.twoProd(b.hl(i,j),a(i,1));
                                        [p3,e3] = Calc.twoProd(b.lh(i,j),a(i,1));
                                        p4 = b.ll(i,j)*a(i,1);
                                        [p2,e1] = Calc.twoSum(p2,e1);
                                        [p3,e2,e1] = Calc.threeSum(p3,e2,e1);
                                        [p4,e2] = Calc.threeSum2(p4,e3,e2);
                                        p5 = e1+e2;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            else
                                
                            end
                        else
                            disp('Matrix dimensions must agree');
                            obj=0;
                        end
                    elseif is2 %na==1
                        if ma==nb||ma==mb
                            obj=QD(zeros(nb,mb));
                            if isaQD&&isbQD
                                for i=1:nb
                                    for j=1:mb
                                        [p1,e1] = Calc.twoProd(a.hh(1,j),b.hh(i,j));
                                        [p2,e2] = Calc.twoProd(a.hh(1,j),b.hl(i,j));
                                        [p3,e3] = Calc.twoProd(a.hl(1,j),b.hh(i,j));
                                        [p4,e4] = Calc.twoProd(a.hh(1,j),b.lh(i,j));
                                        [p5,e5] = Calc.twoProd(a.hl(1,j),b.hl(i,j));
                                        [p6,e6] = Calc.twoProd(a.lh(1,j),b.hh(i,j));
                                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                        [p4,p5,p6] = Calc.threeSum(p4,p5,p6);
                                        [p3,e2,e3] = Calc.threeSum(p3,e2,e3);
                                        [p3,p4] = Calc.twoSum(p3,p4);
                                        [p5,e2] = Calc.twoSum(e2,p5);
                                        [p4,p5] = Calc.twoSum(p4,p5);
                                        p5 = p5+e2+e3+p6;
                                        p4 = p4+a.hh(1,j)*b.ll(i,j)+a.hl(1,j)*b.lh(i,j)+a.lh(1,j)*b.hl(i,j)+a.ll(1,j)*b.hh(i,j)+e1+e4+e5+e6;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaQD&&isbDD
                                for i=1:nb
                                    for j=1:mb
                                        [p1,e1] = Calc.twoProd(a.hh(1,j),b.hi(i,j));
                                        [p2,e2] = Calc.twoProd(a.hh(1,j),b.lo(i,j));
                                        [p3,e3] = Calc.twoProd(a.hl(1,j),b.hi(i,j));
                                        [p4,e4] = Calc.twoProd(a.hl(1,j),b.lo(i,j));
                                        [p5,e5] = Calc.twoProd(a.lh(1,j),b.hi(i,j));
                                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                        [p3,p4,p5] = Calc.threeSum(p3,p4,p5);
                                        [e2,e3] = Calc.twoSum(e2,e3);
                                        [p3,e2] = Calc.twoSum(p3,e2);
                                        [p4,e3] = Calc.twoSum(p4,e3);
                                        [p4,e2] = Calc.twoSum(p4,e2);
                                        p5 = p5+e2+e3;
                                        s = a.lh(1,j)*b.lo(i,j) + a.ll(1,j)*b.hi(i,j) + e4 + e5;
                                        [p4,e1] = Calc.threeSum2(p4,e1,s);
                                        p5 = p5+e1;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaDD&&isbQD
                                for i=1:nb
                                    for j=1:mb
                                        [p1,e1] = Calc.twoProd(b.hh(i,j),a.hi(1,j));
                                        [p2,e2] = Calc.twoProd(b.hh(i,j),a.lo(1,j));
                                        [p3,e3] = Calc.twoProd(b.hl(i,j),a.hi(1,j));
                                        [p4,e4] = Calc.twoProd(b.hl(i,j),a.lo(1,j));
                                        [p5,e5] = Calc.twoProd(b.lh(i,j),a.hi(1,j));
                                        [p2,p3,e1] = Calc.threeSum(p2,p3,e1);
                                        [p3,p4,p5] = Calc.threeSum(p3,p4,p5);
                                        [e2,e3] = Calc.twoSum(e2,e3);
                                        [p3,e2] = Calc.twoSum(p3,e2);
                                        [p4,e3] = Calc.twoSum(p4,e3);
                                        [p4,e2] = Calc.twoSum(p4,e2);
                                        p5 = p5+e2+e3;
                                        s = b.lh(i,j)*a.lo(1,j) + b.ll(i,j)*a.hi(1,j) + e4 + e5;
                                        [p4,e1] = Calc.threeSum2(p4,e1,s);
                                        p5 = p5+e1;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaQD&&isbD
                                for i=1:nb
                                    for j=1:mb
                                        [p1,e1] = Calc.twoProd(a.hh(1,j),b(i,j));
                                        [p2,e2] = Calc.twoProd(a.hl(1,j),b(i,j));
                                        [p3,e3] = Calc.twoProd(a.lh(1,j),b(i,j));
                                        p4 = a.ll(1,j)*b(i,j);
                                        [p2,e1] = Calc.twoSum(p2,e1);
                                        [p3,e2,e1] = Calc.threeSum(p3,e2,e1);
                                        [p4,e2] = Calc.threeSum2(p4,e3,e2);
                                        p5 = e1+e2;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            elseif isaD&&isbQD
                                for i=1:nb
                                    for j=1:mb
                                        [p1,e1] = Calc.twoProd(b.hh(i,j),a(1,j));
                                        [p2,e2] = Calc.twoProd(b.hl(i,j),a(1,j));
                                        [p3,e3] = Calc.twoProd(b.lh(i,j),a(1,j));
                                        p4 = b.ll(i,j)*a(1,j);
                                        [p2,e1] = Calc.twoSum(p2,e1);
                                        [p3,e2,e1] = Calc.threeSum(p3,e2,e1);
                                        [p4,e2] = Calc.threeSum2(p4,e3,e2);
                                        p5 = e1+e2;
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(p1,p2,p3,p4,p5);
                                    end
                                end
                            else
                                
                            end
                        else
                            disp('Matrix dimensions must agree');
                            obj=0;
                        end
                    end
                end
            end
        end
        %a/b
        function obj=mrdivide(a, b)
            obj=QD(0);
            [m2, n2] = size(b);
            if(m2 ~= 1 || n2 ~= 1)
                %unsupported operation
                abort();
            end
            isN1 = isnumeric(a);
            isN2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            isQD1 = isa(a, 'QD');
            isQD2 = isa(b, 'QD');
            
            if(isN1)
                if(isQD2)%double/QD
                    q1 = a/b.hh;     r = minus(a,mtimes(b,q1));
                    q2 = r.hh/b.hh;  r = minus(r,mtimes(b,q2));
                    q3 = r.hh/b.hh;  r = minus(r,mtimes(b,q3));
                    q4 = r.hh/b.hh;  r = minus(r,mtimes(b,q4));
                    q5 = r.hh/b.hh;
                    [obj.hh,obj.hl,obj.lh,obj.ll] = Calc.renormalize(q1,q2,q3,q4,q5);
                else
                    %obj2 is not supported
                end
            elseif(isDD1)
                if(isQD2)%DD/QD
                    q1 = a.hi/b.hh;  r = minus(a,mtimes(b,q1));
                    q2 = r.hh/b.hh;  r = minus(r,mtimes(b,q2));
                    q3 = r.hh/b.hh;  r = minus(r,mtimes(b,q3));
                    q4 = r.hh/b.hh;  r = minus(r,mtimes(b,q4));
                    q5 = r.hh/b.hh;
                    [obj.hh,obj.hl,obj.lh,obj.ll] = Calc.renormalize(q1,q2,q3,q4,q5);
                end
            elseif(isQD1)
                if(isN2)%QD/double
                    q1 = a.hh/b;
                    [th,tl] = Calc.twoProd(q1,b);
                    t=DD(th,tl);
                    r = minus(a,t);
                    q2 = r.hh/b;
                    [th,tl] = Calc.twoProd(q2,b);
                    t=DD(th,tl);
                    r = minus(r,t);
                    q3 = r.hh/b;
                    [th,tl] = Calc.twoProd(q3,b);
                    t=DD(th,tl);
                    r = minus(r,t);
                    q4 = r.hh/b;
                    [th,tl] = Calc.twoProd(q4,b);
                    t=DD(th,tl);
                    r = minus(r,t);
                    q5 = r.hh/b;
                    [obj.hh,obj.hl,obj.lh,obj.ll]= Calc.renormalize(q1,q2,q3,q4,q5);
                elseif(isDD2)%QD/DD
                    q1 = a.hh/b.hi;  r = minus(a,mtimes(b,q1));
                    q2 = r.hh/b.hi;  r = minus(r,mtimes(b,q2));
                    q3 = r.hh/b.hi;  r = minus(r,mtimes(b,q3));
                    q4 = r.hh/b.hi;  r = minus(r,mtimes(b,q4));
                    q5 = r.hh/b.hi;
                    [obj.hh,obj.hl,obj.lh,obj.ll] = Calc.renormalize(q1,q2,q3,q4,q5);
                elseif(isQD2)%QD/QD
                    q1 = a.hh/b.hh;  r = minus(a,mtimes(b,q1));
                    q2 = r.hh/b.hh;  r = minus(r,mtimes(b,q2));
                    q3 = r.hh/b.hh;  r = minus(r,mtimes(b,q3));
                    q4 = r.hh/b.hh;
                    [obj.hh,obj.hl,obj.lh,obj.ll] = Calc.renormalize(q1,q2,q3,q4,0);
                else
                    %obj2 is not supported
                end
            else
                %obj1 is not supported
            end
            
        end
        
        %a\b
        function obj=mldivide(a,b)
            obj=mrdivide(b,a);
        end
        
        %a./b
        function obj=rdivide(a,b)
            [na,ma]=size(a);
            [nb,mb]=size(b);
            is1=na~=1;
            is2=ma~=1;
            is3=nb~=1;
            is4=mb~=1;
            isaDD=isa(a,'DD');
            isaQD=isa(a,'QD');
            isbDD=isa(b,'DD');
            isbQD=isa(b,'QD');
            isaD=isnumeric(a);
            isbD=isnumeric(b);
            if isaD&&isbD
            elseif isaD&&~isbD
                warning('off','MATLAB:structOnObject');
                b=struct(b);
            elseif ~isaD&&isbD
                warning('off','MATLAB:structOnObject');
                a=struct(a);
            else
                warning('off','MATLAB:structOnObject');
                a=struct(a);
                b=struct(b);
            end
            if (is1&&is2&&is3&&is4)%not vector
                if (nb~=na)||(ma~=mb)
                    disp('Matrix dimensions must agree');
                    obj=0;
                else
                    obj=QD(zeros(na,mb));
                    if isaQD&&isbQD %QD,QD
                        for i=1:na
                            for j=1:mb
                                q1 = a.hh(i,j)/b.hh(i,j);  r = minus(QD(a.hh(i,j),a.hl(i,j),a.lh(i,j),a.ll(i,j)),mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q1));
                                q2 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q2));
                                q3 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q3));
                                q4 = r.hh/b.hh(i,j);
                                [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,0);
                            end
                        end
                    elseif isaQD&&isbDD %QD,DD
                        for i=1:na
                            for j=1:mb
                                q1 = a.hh(i,j)/b.hi(i,j);  r = minus(QD(a.hh(i,j),a.hl(i,j),a.lh(i,j),a.ll(i,j)),mtimes(DD(b.hi(i,j),b.lo(i,j)),q1));
                                q2 = r.hh/b.hi(i,j);  r = minus(r,mtimes(DD(b.hi(i,j),b.lo(i,j)),q2));
                                q3 = r.hh/b.hi(i,j);  r = minus(r,mtimes(DD(b.hi(i,j),b.lo(i,j)),q3));
                                q4 = r.hh/b.hi(i,j);  r = minus(r,mtimes(DD(b.hi(i,j),b.lo(i,j)),q4));
                                q5 = r.hh/b.hi(i,j);
                                [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                            end
                        end
                    elseif isaDD&&isbQD%DD,QD
                        for i=1:na
                            for j=1:mb
                                q1 = a.hi(i,j)/b.hh(i,j);  r = minus(DD(a.hi(i,j),a.lo(i,j)),mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q1));
                                q2 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q2));
                                q3 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q3));
                                q4 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q4));
                                q5 = r.hh/b.hh(i,j);
                                [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                            end
                        end
                    elseif isaQD&&isbD%QD,D
                        for i=1:na
                            for j=1:mb
                                q1 = a.hh(i,j)/b(i,j);
                                [th,tl] = Calc.twoProd(q1,b(i,j));
                                t=DD(th,tl);
                                r = minus(QD(a.hh(i,j),a.hl(i,j),a.lh(i,j),a.ll(i,j)),t);
                                q2 = r.hh/b(i,j);
                                [th,tl] = Calc.twoProd(q2,b(i,j));
                                t=DD(th,tl);
                                r = minus(r,t);
                                q3 = r.hh/b(i,j);
                                [th,tl] = Calc.twoProd(q3,b(i,j));
                                t=DD(th,tl);
                                r = minus(r,t);
                                q4 = r.hh/b(i,j);
                                [th,tl] = Calc.twoProd(q4,b(i,j));
                                t=DD(th,tl);
                                r = minus(r,t);
                                q5 = r.hh/b(i,j);
                                [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)]= Calc.renormalize(q1,q2,q3,q4,q5);
                            end
                        end
                    elseif isaD&&isbQD%D,QD
                        for i=1:na
                            for j=1:mb
                                q1 = a(i,j)/b.hh(i,j);     r = minus(a(i,j),mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q1));
                                q2 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q2));
                                q3 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q3));
                                q4 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q4));
                                q5 = r.hh/b.hh(i,j);
                                [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                            end
                        end
                    else
                        
                    end
                end
            else %include vector
                if (is1)&&(is2)
                    if is3 %mb==1
                        if nb==na||nb==ma
                            obj=QD(zeros(na,ma));
                            if isaQD&&isbQD
                                for i=1:na
                                    for j=1:ma
                                        q1 = a.hh(i,j)/b.hh(i,1);  r = minus(QD(a.hh(i,j),a.hl(i,j),a.lh(i,j),a.ll(i,j)),mtimes(QD(b.hh(i,1),b.hl(i,1),b.lh(i,1),b.ll(i,1)),q1));
                                        q2 = r.hh/b.hh(i,1);  r = minus(r,mtimes(QD(b.hh(i,1),b.hl(i,1),b.lh(i,1),b.ll(i,1)),q2));
                                        q3 = r.hh/b.hh(i,1);  r = minus(r,mtimes(QD(b.hh(i,1),b.hl(i,1),b.lh(i,1),b.ll(i,1)),q3));
                                        q4 = r.hh/b.hh(i,1);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,0);
                                    end
                                end
                            elseif isaQD&&isbDD
                                for i=1:na
                                    for j=1:ma
                                        q1 = a.hh(i,j)/b.hi(i,1);  r = minus(QD(a.hh(i,j),a.hl(i,j),a.lh(i,j),a.ll(i,j)),mtimes(DD(b.hi(i,1),b.lo(i,1)),q1));
                                        q2 = r.hh/b.hi(i,1);  r = minus(r,mtimes(DD(b.hi(i,1),b.lo(i,1)),q2));
                                        q3 = r.hh/b.hi(i,1);  r = minus(r,mtimes(DD(b.hi(i,1),b.lo(i,1)),q3));
                                        q4 = r.hh/b.hi(i,1);  r = minus(r,mtimes(DD(b.hi(i,1),b.lo(i,1)),q4));
                                        q5 = r.hh/b.hi(i,1);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            elseif isaDD&&isbQD
                                for i=1:na
                                    for j=1:ma
                                        q1 = a.hi(i,j)/b.hh(i,1);  r = minus(DD(a.hi(i,j),a.lo(i,j)),mtimes(QD(b.hh(i,1),b.hl(i,1),b.lh(i,1),b.ll(i,1)),q1));
                                        q2 = r.hh/b.hh(i,1);  r = minus(r,mtimes(QD(b.hh(i,1),b.hl(i,1),b.lh(i,1),b.ll(i,1)),q2));
                                        q3 = r.hh/b.hh(i,1);  r = minus(r,mtimes(QD(b.hh(i,1),b.hl(i,1),b.lh(i,1),b.ll(i,1)),q3));
                                        q4 = r.hh/b.hh(i,1);  r = minus(r,mtimes(QD(b.hh(i,1),b.hl(i,1),b.lh(i,1),b.ll(i,1)),q4));
                                        q5 = r.hh/b.hh(i,1);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            elseif isaQD&&isbD
                                for i=1:na
                                    for j=1:ma
                                        q1 = a.hh(i,j)/b(i,1);
                                        [th,tl] = Calc.twoProd(q1,b(i,1));
                                        t=DD(th,tl);
                                        r = minus(QD(a.hh(i,j),a.hl(i,j),a.lh(i,j),a.ll(i,j)),t);
                                        q2 = r.hh/b(i,1);
                                        [th,tl] = Calc.twoProd(q2,b(i,1));
                                        t=DD(th,tl);
                                        r = minus(r,t);
                                        q3 = r.hh/b(i,1);
                                        [th,tl] = Calc.twoProd(q3,b(i,1));
                                        t=DD(th,tl);
                                        r = minus(r,t);
                                        q4 = r.hh/b(i,1);
                                        [th,tl] = Calc.twoProd(q4,b(i,1));
                                        t=DD(th,tl);
                                        r = minus(r,t);
                                        q5 = r.hh/b(i,1);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)]= Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            elseif isaD&&isbQD
                                for i=1:na
                                    for j=1:ma
                                        q1 = a(i,j)/b.hh(i,1);     r = minus(a(i,j),mtimes(QD(b.hh(i,1),b.hl(i,1),b.lh(i,1),b.ll(i,1)),q1));
                                        q2 = r.hh/b.hh(i,1);  r = minus(r,mtimes(QD(b.hh(i,1),b.hl(i,1),b.lh(i,1),b.ll(i,1)),q2));
                                        q3 = r.hh/b.hh(i,1);  r = minus(r,mtimes(QD(b.hh(i,1),b.hl(i,1),b.lh(i,1),b.ll(i,1)),q3));
                                        q4 = r.hh/b.hh(i,1);  r = minus(r,mtimes(QD(b.hh(i,1),b.hl(i,1),b.lh(i,1),b.ll(i,1)),q4));
                                        q5 = r.hh/b.hh(i,1);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            else
                                
                            end
                        else
                            disp('Matrix dimensions must agree');
                            obj=0;
                        end
                    elseif is4 %nb==1
                        if mb==na||mb==ma
                            obj=QD(zeros(na,ma));
                            if isaQD&&isbQD
                                for i=1:na
                                    for j=1:ma
                                        q1 = a.hh(i,j)/b.hh(1,j);  r = minus(QD(a.hh(i,j),a.hl(i,j),a.lh(i,j),a.ll(i,j)),mtimes(QD(b.hh(1,j),b.hl(1,j),b.lh(1,j),b.ll(1,j)),q1));
                                        q2 = r.hh/b.hh(1,j);  r = minus(r,mtimes(QD(b.hh(1,j),b.hl(1,j),b.lh(1,j),b.ll(1,j)),q2));
                                        q3 = r.hh/b.hh(1,j);  r = minus(r,mtimes(QD(b.hh(1,j),b.hl(1,j),b.lh(1,j),b.ll(1,j)),q3));
                                        q4 = r.hh/b.hh(1,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,0);
                                    end
                                end
                                
                            elseif isaQD&&isbDD
                                for i=1:na
                                    for j=1:ma
                                        q1 = a.hh(i,j)/b.hi(1,j);  r = minus(QD(a.hh(i,j),a.hl(i,j),a.lh(i,j),a.ll(i,j)),mtimes(DD(b.hi(1,j),b.lo(1,j)),q1));
                                        q2 = r.hh/b.hi(1,j);  r = minus(r,mtimes(DD(b.hi(1,j),b.lo(1,j)),q2));
                                        q3 = r.hh/b.hi(1,j);  r = minus(r,mtimes(DD(b.hi(1,j),b.lo(1,j)),q3));
                                        q4 = r.hh/b.hi(1,j);  r = minus(r,mtimes(DD(b.hi(1,j),b.lo(1,j)),q4));
                                        q5 = r.hh/b.hi(1,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            elseif isaDD&&isbQD
                                for i=1:na
                                    for j=1:ma
                                        q1 = a.hi(i,j)/b.hh(1,j);  r = minus(DD(a.hi(i,j),a.lo(i,j)),mtimes(QD(b.hh(1,j),b.hl(1,j),b.lh(1,j),b.ll(1,j)),q1));
                                        q2 = r.hh/b.hh(1,j);  r = minus(r,mtimes(QD(b.hh(1,j),b.hl(1,j),b.lh(1,j),b.ll(1,j)),q2));
                                        q3 = r.hh/b.hh(1,j);  r = minus(r,mtimes(QD(b.hh(1,j),b.hl(1,j),b.lh(1,j),b.ll(1,j)),q3));
                                        q4 = r.hh/b.hh(1,j);  r = minus(r,mtimes(QD(b.hh(1,j),b.hl(1,j),b.lh(1,j),b.ll(1,j)),q4));
                                        q5 = r.hh/b.hh(1,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            elseif isaQD&&isbD
                                for i=1:na
                                    for j=1:ma
                                        q1 = a.hh(i,j)/b(1,j);
                                        [th,tl] = Calc.twoProd(q1,b(1,j));
                                        t=DD(th,tl);
                                        r = minus(QD(a.hh(i,j),a.hl(i,j),a.lh(i,j),a.ll(i,j)),t);
                                        q2 = r.hh/b(1,j);
                                        [th,tl] = Calc.twoProd(q2,b(1,j));
                                        t=DD(th,tl);
                                        r = minus(r,t);
                                        q3 = r.hh/b(1,j);
                                        [th,tl] = Calc.twoProd(q3,b(1,j));
                                        t=DD(th,tl);
                                        r = minus(r,t);
                                        q4 = r.hh/b(1,j);
                                        [th,tl] = Calc.twoProd(q4,b(1,j));
                                        t=DD(th,tl);
                                        r = minus(r,t);
                                        q5 = r.hh/b(1,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)]= Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            elseif isaD&&isbQD
                                for i=1:na
                                    for j=1:ma
                                        q1 = a(i,j)/b.hh(1,j);     r = minus(a(i,j),mtimes(QD(b.hh(1,j),b.hl(1,j),b.lh(1,j),b.ll(1,j)),q1));
                                        q2 = r.hh/b.hh(1,j);  r = minus(r,mtimes(QD(b.hh(1,j),b.hl(1,j),b.lh(1,j),b.ll(1,j)),q2));
                                        q3 = r.hh/b.hh(1,j);  r = minus(r,mtimes(QD(b.hh(1,j),b.hl(1,j),b.lh(1,j),b.ll(1,j)),q3));
                                        q4 = r.hh/b.hh(1,j);  r = minus(r,mtimes(QD(b.hh(1,j),b.hl(1,j),b.lh(1,j),b.ll(1,j)),q4));
                                        q5 = r.hh/b.hh(1,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                                        
                                    end
                                end
                            else
                                
                            end
                        else
                            disp('Matrix dimensions must agree');
                            obj=0;
                        end
                    end
                else
                    if is1 %ma==1
                        if nb==na||nb==ma
                            obj=QD(zeros(nb,mb));
                            if isaQD&&isbQD
                                for i=1:nb
                                    for j=1:mb
                                        q1 = a.hh(i,1)/b.hh(i,j);  r = minus(QD(a.hh(i,1),a.hl(i,1),a.lh(i,1),a.ll(i,1)),mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q1));
                                        q2 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q2));
                                        q3 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q3));
                                        q4 = r.hh/b.hh(i,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,0);
                                    end
                                end
                            elseif isaQD&&isbDD %QD,DD
                                for i=1:nb
                                    for j=1:mb
                                        q1 = a.hh(i,1)/b.hi(i,j);  r = minus(QD(a.hh(i,1),a.hl(i,1),a.lh(i,1),a.ll(i,1)),mtimes(DD(b.hi(i,j),b.lo(i,j)),q1));
                                        q2 = r.hh/b.hi(i,j);  r = minus(r,mtimes(DD(b.hi(i,j),b.lo(i,j)),q2));
                                        q3 = r.hh/b.hi(i,j);  r = minus(r,mtimes(DD(b.hi(i,j),b.lo(i,j)),q3));
                                        q4 = r.hh/b.hi(i,j);  r = minus(r,mtimes(DD(b.hi(i,j),b.lo(i,j)),q4));
                                        q5 = r.hh/b.hi(i,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            elseif isaDD&&isbQD%DD,QD
                                for i=1:nb
                                    for j=1:mb
                                        q1 = a.hi(i,1)/b.hh(i,j);  r = minus(DD(a.hi(i,1),a.lo(i,1)),mtimes(QD(b.hh(i,1),b.hl(i,1),b.lh(i,j),b.ll(i,j)),q1));
                                        q2 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q2));
                                        q3 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q3));
                                        q4 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q4));
                                        q5 = r.hh/b.hh(i,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            elseif isaQD&&isbD%QD,D
                                for i=1:nb
                                    for j=1:mb
                                        q1 = a.hh(i,1)/b(i,j);
                                        [th,tl] = Calc.twoProd(q1,b(i,j));
                                        t=DD(th,tl);
                                        r = minus(QD(a.hh(i,1),a.hl(i,1),a.lh(i,1),a.ll(i,1)),t);
                                        q2 = r.hh/b(i,j);
                                        [th,tl] = Calc.twoProd(q2,b(i,j));
                                        t=DD(th,tl);
                                        r = minus(r,t);
                                        q3 = r.hh/b(i,j);
                                        [th,tl] = Calc.twoProd(q3,b(i,j));
                                        t=DD(th,tl);
                                        r = minus(r,t);
                                        q4 = r.hh/b(i,j);
                                        [th,tl] = Calc.twoProd(q4,b(i,j));
                                        t=DD(th,tl);
                                        r = minus(r,t);
                                        q5 = r.hh/b(i,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)]= Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            elseif isaD&&isbQD%D,QD
                                for i=1:nb
                                    for j=1:mb
                                        q1 = a(i,1)/b.hh(i,j);     r = minus(a(i,1),mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q1));
                                        q2 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q2));
                                        q3 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q3));
                                        q4 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q4));
                                        q5 = r.hh/b.hh(i,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            else
                                
                            end
                        else
                            disp('Matrix dimensions must agree');
                            obj=0;
                        end
                    elseif is2 %na==1
                        if ma==nb||ma==mb
                            obj=QD(zeros(nb,mb));
                            if isaQD&&isbQD
                                for i=1:nb
                                    for j=1:mb
                                        q1 = a.hh(1,j)/b.hh(i,j);  r = minus(QD(a.hh(1,j),a.hl(1,j),a.lh(1,j),a.ll(1,j)),mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q1));
                                        q2 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q2));
                                        q3 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q3));
                                        q4 = r.hh/b.hh(i,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,0);
                                    end
                                end
                            elseif isaQD&&isbDD %QD,DD
                                for i=1:nb
                                    for j=1:mb
                                        q1 = a.hh(1,j)/b.hi(i,j);  r = minus(QD(a.hh(1,j),a.hl(1,j),a.lh(1,j),a.ll(1,j)),mtimes(DD(b.hi(i,j),b.lo(i,j)),q1));
                                        q2 = r.hh/b.hi(i,j);  r = minus(r,mtimes(DD(b.hi(i,j),b.lo(i,j)),q2));
                                        q3 = r.hh/b.hi(i,j);  r = minus(r,mtimes(DD(b.hi(i,j),b.lo(i,j)),q3));
                                        q4 = r.hh/b.hi(i,j);  r = minus(r,mtimes(DD(b.hi(i,j),b.lo(i,j)),q4));
                                        q5 = r.hh/b.hi(i,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            elseif isaDD&&isbQD%DD,QD
                                for i=1:nb
                                    for j=1:mb
                                        q1 = a.hi(1,j)/b.hh(i,j);  r = minus(DD(a.hi(1,j),a.lo(1,j)),mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q1));
                                        q2 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q2));
                                        q3 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q3));
                                        q4 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q4));
                                        q5 = r.hh/b.hh(i,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            elseif isaQD&&isbD%QD,D
                                for i=1:nb
                                    for j=1:mb
                                        q1 = a.hh(1,j)/b(i,j);
                                        [th,tl] = Calc.twoProd(q1,b(i,j));
                                        t=DD(th,tl);
                                        r = minus(QD(a.hh(1,j),a.hl(1,j),a.lh(1,j),a.ll(1,j)),t);
                                        q2 = r.hh/b(i,j);
                                        [th,tl] = Calc.twoProd(q2,b(i,j));
                                        t=DD(th,tl);
                                        r = minus(r,t);
                                        q3 = r.hh/b(i,j);
                                        [th,tl] = Calc.twoProd(q3,b(i,j));
                                        t=DD(th,tl);
                                        r = minus(r,t);
                                        q4 = r.hh/b(i,j);
                                        [th,tl] = Calc.twoProd(q4,b(i,j));
                                        t=DD(th,tl);
                                        r = minus(r,t);
                                        q5 = r.hh/b(i,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)]= Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            elseif isaD&&isbQD%D,QD
                                for i=1:nb
                                    for j=1:mb
                                        q1 = a(1,j)/b.hh(i,j);     r = minus(a(1,j),mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q1));
                                        q2 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q2));
                                        q3 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q3));
                                        q4 = r.hh/b.hh(i,j);  r = minus(r,mtimes(QD(b.hh(i,j),b.hl(i,j),b.lh(i,j),b.ll(i,j)),q4));
                                        q5 = r.hh/b.hh(i,j);
                                        [obj.hh(i,j),obj.hl(i,j),obj.lh(i,j),obj.ll(i,j)] = Calc.renormalize(q1,q2,q3,q4,q5);
                                    end
                                end
                            else
                                
                            end
                        else
                            disp('Matrix dimensions must agree');
                            obj=0;
                        end
                    end
                end
            end
        end
        
        %a.\b
        function obj=ldivide(a,b)
            obj=rdivide(b,a);
        end
        
        %----------------%
        %    booleans    %
        %----------------%
        %a<b
        function obj=lt(a,b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            if ~(m1==m2&&n1==n2)
                disp('matrix dimension must agree');
            end
            obj=zeros(m1,n1);
            isN1 = isnumeric(a);
            isN2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            isQD1 = isa(a, 'QD');
            isQD2 = isa(b, 'QD');
            
            
            if(isN1)
                if(isQD2)%double<QD
                    warning('off','MATLAB:structOnObject');
                    
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a(i, j) == b.hh(i, j) && b.hl(i,j)>0
                                obj(i,j) =true;
                            elseif a(i,j) < b.hh(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                else
                    
                end
            elseif(isDD1)
                if(isQD2)%DD<QD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hi(i,j)<b.hh(i,j)
                                obj(i,j)=true;
                            elseif a.hi(i,j)==b.hh(i,j) && a.lo(i,j)<b.hl(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                end
            elseif(isQD1)
                if(isN2)%QD<D
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)<b(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b(i,j) && a.hl(i,j)<0
                                obj(i,j)=true;
                            end
                        end
                    end
                elseif(isDD2)%QD<DD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)<b.hi(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b.hi(i,j) && a.hl(i,j)<b.lo(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                elseif(isQD2)%QD<QD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)<b.hh(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)<b.hh(i,j) && a.hl(i,j)<b.hl(i,j)
                                obj(i,j)=true;
                            elseif  a.hh(i,j)==b.hh(i,j) && a.hl(i,j)==b.hl(i,j) && a.lh(i,j)<b.lh(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b.hh(i,j) && a.hl(i,j)==b.hl(i,j) && a.lh(i,j)==b.lh(i,j) && a.ll(i,j)<b.ll(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                    
                else
                    
                end
            else
                
            end
            obj=logical(obj);
        end
        
        %a>b
        function obj=gt(a,b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            if ~(m1==m2&&n1==n2)
                disp('matrix dimension must agree');
            end
            obj=zeros(m1,n1);
            isN1 = isnumeric(a);
            isN2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            isQD1 = isa(a, 'QD');
            isQD2 = isa(b, 'QD');
            
            
            if(isN1)
                if(isQD2)%double>QD
                    warning('off','MATLAB:structOnObject');
                    
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a(i,j)>b.hh(i,j)
                                obj(i,j)=true;
                            elseif a(i,j)==b.hh(i,j) && b.hl(i,j)<0
                                obj(i,j)=true;
                            end
                        end
                    end
                else
                    
                end
            elseif(isDD1)
                if(isQD2)%DD>QD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hi(i,j)>b.hh(i,j)
                                obj(i,j)=true;
                            elseif a.hi(i,j)==b.hh(i,j) && a.lo(i,j)>b.hl(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                end
            elseif(isQD1)
                if(isN2)%QD>D
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)>b(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b(i,j) && a.hl(i,j)>0
                                obj(i,j)=true;
                            end
                        end
                    end
                elseif(isDD2)%QD>DD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)>b.hi(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b.hi(i,j) && a.hl(i,j)>b.lo(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                elseif(isQD2)%QD>QD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)>b.hh(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b.hh(i,j) && a.hl(i,j)>b.hl(i,j)
                                obj(i,j)=true;
                            elseif  a.hh(i,j)==b.hh(i,j) && a.hl(i,j)==b.hl(i,j) && a.lh(i,j)>b.lh(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b.hh(i,j) && a.hl(i,j)==b.hl(i,j) && a.lh(i,j)==b.lh(i,j) && a.ll(i,j)>b.ll(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                    
                else
                    
                end
            else
                
            end
            obj=logical(obj);
        end
        
        %a<=b
        function obj=le(a,b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            if ~(m1==m2&&n1==n2)
                disp('matrix dimension must agree');
            end
            obj=zeros(m1,n1);
            isN1 = isnumeric(a);
            isN2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            isQD1 = isa(a, 'QD');
            isQD2 = isa(b, 'QD');
            
            
            if(isN1)
                if(isQD2)%double<=QD
                    warning('off','MATLAB:structOnObject');
                    
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a(i,j)<b.hh(i,j)
                                obj(i,j)=true;
                            elseif a(i,j)==b.hh(i,j) && b.hl(i,j)>=0
                                obj(i,j)=true;
                            end
                        end
                    end
                else
                    
                end
            elseif(isDD1)
                if(isQD2)%DD<=QD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hi(i,j)<b.hh(i,j)
                                obj(i,j)=true;
                            elseif a.hi(i,j)==b.hh(i,j) && a.lo(i,j)<=b.hl(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                end
            elseif(isQD1)
                if(isN2)%QD<=D
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)<=b(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b(i,j) && a.hl(i,j)<=0
                                obj(i,j)=true;
                            end
                        end
                    end
                elseif(isDD2)%QD<=DD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)<=b.hi(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b.hi(i,j) && a.hl(i,j)<=b.lo(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                elseif(isQD2)%QD<=QD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)<=b.hh(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)<=b.hh(i,j) && a.hl(i,j)<=b.hl(i,j)
                                obj(i,j)=true;
                            elseif  a.hh(i,j)==b.hh(i,j) && a.hl(i,j)==b.hl(i,j) && a.lh(i,j)<=b.lh(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b.hh(i,j) && a.hl(i,j)==b.hl(i,j) && a.lh(i,j)==b.lh(i,j) && a.ll(i,j)<=b.ll(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                    
                else
                    
                end
            else
                
            end
            obj=logical(obj);
        end
        
        %a>=b
        function obj=ge(a,b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            if ~(m1==m2&&n1==n2)
                disp('matrix dimension must agree');
            end
            obj=zeros(m1,n1);
            isN1 = isnumeric(a);
            isN2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            isQD1 = isa(a, 'QD');
            isQD2 = isa(b, 'QD');
            
            
            if(isN1)
                if(isQD2)%double>=QD
                    warning('off','MATLAB:structOnObject');
                    
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a(i,j)>=b.hh(i,j)
                                obj(i,j)=true;
                            elseif a(i,j)==b.hh(i,j) && b.hl(i,j)<=0
                                obj(i,j)=true;
                            end
                        end
                    end
                else
                    
                end
            elseif(isDD1)
                if(isQD2)%DD>=QD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hi(i,j)>=b.hh(i,j)
                                obj(i,j)=true;
                            elseif a.hi(i,j)==b.hh(i,j) && a.lo(i,j)>=b.hl(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                end
            elseif(isQD1)
                if(isN2)%QD>=D
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)>=b(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b(i,j) && a.hl(i,j)>=0
                                obj(i,j)=true;
                            end
                        end
                    end
                elseif(isDD2)%QD>=DD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)>=b.hi(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b.hi(i,j) && a.hl(i,j)>=b.lo(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                elseif(isQD2)%QD>=QD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)>=b.hh(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)>=b.hh(i,j) && a.hl(i,j)>=b.hl(i,j)
                                obj(i,j)=true;
                            elseif  a.hh(i,j)==b.hh(i,j) && a.hl(i,j)==b.hl(i,j) && a.lh(i,j)>=b.lh(i,j)
                                obj(i,j)=true;
                            elseif a.hh(i,j)==b.hh(i,j) && a.hl(i,j)==b.hl(i,j) && a.lh(i,j)==b.lh(i,j) && a.ll(i,j)>=b.ll(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                    
                else
                    
                end
            else
                
            end
            obj=logical(obj);
        end
        
        %a~=b
        function obj=ne(a,b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            if ~(m1==m2&&n1==n2)
                disp('matrix dimension must agree');
            end
            c = eq(a,b);
            obj = ~c;
        end
        
        %a==b
        function obj=eq(a,b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            if ~(m1==m2&&n1==n2)
                disp('matrix dimension must agree');
            end
            obj=zeros(m1,n1);
            isN1 = isnumeric(a);
            isN2 = isnumeric(b);
            isDD1 = isa(a, 'DD');
            isDD2 = isa(b, 'DD');
            isQD1 = isa(a, 'QD');
            isQD2 = isa(b, 'QD');
            
            if(isN1)
                if(isQD2)%double==QD
                    warning('off','MATLAB:structOnObject');
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a(i,j)==b.hh(i,j) && b.hl(i,j)==0 && b.lh(i,j)==0 && b.ll(i,j)==0
                                obj(i,j)=true;
                            end
                        end
                    end
                else
                    
                end
            elseif(isDD1)
                if(isQD2)%DD==QD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hi(i,j)==b.hh(i,j)&&a.lo(i,j)==b.hl(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                end
            elseif(isQD1)
                if(isN2)%QD==D
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)==b(i,j) && b.hl(i,j)==0 && b.lh(i,j)==0 && b.ll(i,j)==0
                                obj(i,j)=true;
                            end
                        end
                    end
                elseif(isDD2)%QD==DD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)==b.hi(i,j)&&a.hl(i,j)==b.lo(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                elseif(isQD2)%QD==QD
                    warning('off','MATLAB:structOnObject');
                    a=struct(a);
                    b=struct(b);
                    for i=1:m1
                        for j=1:n1
                            if a.hh(i,j)==b.hh(i,j)&&a.hl(i,j)==b.hl(i,j)&&a.lh(i,j)==b.lh(i,j)&&a.ll(i,j)==b.ll(i,j)
                                obj(i,j)=true;
                            end
                        end
                    end
                    
                else
                    
                end
            else
                
            end
            obj=logical(obj);
        end
        
        %a&&b
        function obj=and(a,b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            if(m1 ~= m2 ||  n1 ~= n2)
                %unsupported operation
            end
            obj = double(a) && double(b);
        end
        
        %a||b
        function obj=or(a,b)
            [m1, n1] = size(a);
            [m2, n2] = size(b);
            if(m1 ~= m2 ||  n1 ~= n2)
                %unsupported operation
            end
            obj = double(a) || double(b);
        end
        
        %~a
        function obj = not(a)
            obj = ~double(a);
        end
        
        
    end
end
