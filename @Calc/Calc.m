classdef Calc
    methods(Static)
        function [s, e] = twoSum(a, b)
            s = a + b;
            v = s - a;
            e = (a - (s - v)) + (b - v);
        end
        function [s, e] = fastTwoSum(a, b)
            s = a + b;
            e = b - (s - a);
        end
        function [h, m, l] = threeSum(a, b, c)
            [sh, eh] = Calc.twoSum(a, b);
            [h, sh] = Calc.twoSum(sh, c);
            [m, l] = Calc.twoSum(eh, sh);
        end
        function [s, e] = threeSum2(a, b, c)
            [sh, eh] = Calc.twoSum(a, b);
            [s, el] = Calc.twoSum(sh, c);
            e = eh + el;
        end
        function [h, l] = split(a)
            t = 134217729 * a;
            h = t - (t - a);
            l = a - h;
        end
        function [p, e] = twoProd(a, b)
            p = a * b;
            [ah, al] = Calc.split(a);
            [bh, bl] = Calc.split(b);
            
            e = (ah * bh - p) + ah * bl + al * bh + al * bl;
        end
        function [p, e] = twoprod(a, b)%for inner prod
            p = a .* b;
            [ah, al] = Calc.split(a);
            [bh, bl] = Calc.split(b);
            
            e = (ah .* bh - p) + ah .* bl + al .* bh + al .* bl;
        end
        function [p, e] = twoProdFMA(a,b)
            p=a*b;
            e=FMA(a*b-p);
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
    end
end