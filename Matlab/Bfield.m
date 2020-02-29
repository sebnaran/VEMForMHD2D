function [Bx,By] = Bfield(x,y)
    Bx = (sinh(y)-2*sinh(0.5*y))/(2*sinh(0.5));
    By = 1;
end
