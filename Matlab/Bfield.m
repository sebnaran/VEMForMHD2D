function [Bx,By] = Bfield(x,y)
    Bx = (sinh(y)-2*y*sinh(0.5))/(2*sinh(0.5));
    By = 1;
end
