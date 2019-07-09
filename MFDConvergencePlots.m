shape = 'quads';
%shape = 'triangles';
%shape = 'voronoi';

switch shape
    case 'triangles'
%%%%%%%%%%%%%%%%%Triangles

h = [0.10101525445522107, 0.05018856132284956, 0.025141822757713456, 0.012525468249897755,...
     0.006261260829309998];%,0.0031355820733239914,0.0015683308166871686];

ElectricError = [0.25767443763803977, 0.08600016556755646, 0.02541229003884713, 0.006691640143879588,...
    0.0016505878226838334];

MagneticError = [2.565164964978283, 1.437687982179637, 0.8718683347666736, 0.46167647448054944,...
    0.2265250908154283];

secondtitle = 'Triangles';




%Plotting the electric field

figure(1)

clf

loglog(h,ElectricError,'o')
%plot(log(h),log(ElectricError));

hold on


loglog(h,ElectricError)
%loglog(h,10^(1)*h.^(1.8))
hold on
%Pick a basis point for the triangle
xseed = 0.9*h(5)+h(4)*0.1;
yseed = 0.01*0.25+0.001*.75;
%desiredSlope Of triangle
slope = 2;
%Another x point
xnext = h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'r')

text(xseed*0.7+0.3*xnext,ynext+0.001+0.0005,'1')
text(xseed-0.0005,0.5*yseed+0.5*ynext,'2')





title('Triangular Mesh')





%Now the ticks
%The x-axis
leftaxis = 0.01*0.5+.5*0.001;
rightaxis = 0.1+0.01;
upaxis = 0.4;
downaxis = 0.001*0.95+0.05*0.01;

axis([leftaxis rightaxis downaxis upaxis])
xlabel('Mesh Size')

ax = log10(leftaxis);
bx = log10(rightaxis);
tx = ax:(bx-ax)/10:bx;
tx = 10.^(tx);
ay = log10(downaxis);
by = log10(upaxis);
ty = ay:(by-ay)/10:by;
ty = 10.^(ty);

ylabel('Error On Electric Field')

ex = gca;
ex.FontSize =13;


%xticks(tx)
%yticks(ty)
grid on




%Plotting The magnetic Field

figure(2)
clf
loglog(h,MagneticError,'o')
hold on
loglog(h,MagneticError)



hold on

%Pick a basis point for the triangle
xseed = h(4)*0.05+0.95*h(5);
yseed = 0.3;
%desiredSlope Of triangle
slope = 1;
%Another x point
xnext = h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'r')


text(xseed*0.7+0.3*xnext,ynext+0.01+0.01+0.01+0.01,'1')
text(xseed-0.001/1.5,0.5*yseed+0.5*ynext,'1')










title('Triangular Mesh')
xlabel('Mesh Size')
ylabel('Error On Magnetic Field')


%axis([0.01-0.005 0.1+0.01 0.1+0.1 3])

leftaxis = 0.01-0.005;
rightaxis = 0.1+0.01;
downaxis = 0.1+0.1;
upaxis = 3;

axis([leftaxis rightaxis downaxis upaxis])
xlabel('Mesh Size')

ax = log10(leftaxis);
bx = log10(rightaxis);
tx = ax:(bx-ax)/10:bx;
tx = 10.^(tx);
ay = log10(downaxis);
by = log10(upaxis);
ty = ay:(by-ay)/10:by;
ty = 10.^(ty);

ex = gca;
ex.FontSize =13;

%xticks(tx)
%yticks(ty)

grid on





























    case 'quads'
%%%%%%%%%%%%%%%%%%%PerturbedSquares
%h = [0.471405,0.235702,0.122975,0.0614875,0.0310816,0.0155408,0.00779781];
%The above was the older mesh-size

h = [0.16666666666666666, 0.08333333333333333, 0.043478260869565216, 0.021739130434782608,...
     0.010989010989010988];%, 0.005494505494505494,0.0027548209366391185];

ElectricError = [0.114440086555683, 0.04116361358196967, 0.012656547528009151, 0.0032779829796823587,...
    0.0007945280362943563];

MagneticError = [0.8239224365060367, 0.6334513341946489, 0.33529703808140243, 0.16884095035736615,...
    0.08424444123];
                                                      

secondtitle = 'On Perturbed squares';


%Plotting the electric field

figure(1)

clf

loglog(h,ElectricError,'o')
%plot(log(h),log(ElectricError));

hold on


loglog(h,ElectricError)
%loglog(h,10^(1)*h.^(1.8))
hold on
%Pick a basis point for the triangle
xseed = 0.4*(0.0217+0.01);
yseed = 0.0019;
%desiredSlope Of triangle
slope = 2;
%Another x point
xnext = h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'r')

text(xseed*0.7+0.3*xnext,ynext+0.001,'1')
text(xseed-0.001,0.5*yseed+0.5*ynext,'2')





title('Perturbed Quadritateral Mesh')
%axis([0.01 0.25 (0.001+0.0001)/2 .15])
xlabel('Mesh Size')
ylabel('Error On Electric Field')

leftaxis = 0.01;
rightaxis = 0.19;
downaxis = 0.001*.62+0.0001*.38;
upaxis = 0.15;

axis([leftaxis rightaxis downaxis upaxis])


ax = log10(leftaxis);
bx = log10(rightaxis);
tx = ax:(bx-ax)/10:bx;
tx = 10.^(tx);
ay = log10(downaxis);
by = log10(upaxis);
ty = ay:(by-ay)/10:by;
ty = 10.^(ty);



%xticks(tx)
%yticks(ty)


ex = gca;
ex.FontSize =13;


grid on



%Plotting The magnetic Field

figure(2)
clf
loglog(h,MagneticError,'o')
hold on
loglog(h,MagneticError)



hold on

%Pick a basis point for the triangle
xseed = h(4)*0.05+0.95*h(5);
yseed = 0.11;
%desiredSlope Of triangle
slope = 1;
%Another x point
xnext = h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'r')


text(xseed*0.7+0.3*xnext,ynext+0.01+0.001,'1')
text(xseed-0.001,0.5*yseed+0.5*ynext,'1')










title('Perturbed Quadrilateral Mesh')
xlabel('Mesh Size')
ylabel('Error On Magnetic Field')
%axis([0.01 0.25 0.75*10^(-1)+0.25*0.01 1])
grid on

leftaxis = 0.01;
rightaxis = 0.2;
downaxis = 0.75*10^(-1)+0.25*0.01;
upaxis = 1;

axis([leftaxis rightaxis downaxis upaxis])


ax = log10(leftaxis);
bx = log10(rightaxis);
tx = ax:(bx-ax)/10:bx;
tx = 10.^(tx);
ay = log10(downaxis);
by = log10(upaxis);
ty = ay:(by-ay)/10:by;
ty = 10.^(ty);


ex = gca;
ex.FontSize =13;
%xticks(tx)
%yticks(ty)







    case 'voronoi'
%%%%%%%%%%%%%%%%%%%%%%%%%%Voronoi

%h = [0.12803687993289598, 0.06772854614785964, 0.03450327796711771, 0.017476749542968805,...
%     0.008787156237382746];%, 0.004419676355414694,0.0022139558199118672];

%ElectricError = [0.4846671385368301, 0.17629056352803815, 0.0894535222510598, 0.013677866376455412,...
%    0.006616196310031363];

%MagneticError = [4.418879142501644, 2.4900590603987354, 1.2075527434363857, 0.5879063547722386,...
% 0.276556530752596];

h = [0.171499,0.128037,0.0940721,0.0677285,0.0345857,0.0246407];

ElectricError = [0.8528270106417726, 0.4161375289867365, 0.2631698903190134,0.1773556356404628, 0.048334213950323605,...
    0.024966429198926957];

MagneticError = [5.846635716122314, 3.5784520147133114, 3.882480434990292, 2.491257361314027, 1.24380613503858,...
    0.8622777060006687];




secondtitle='Voronoi';



figure(1)

clf

loglog(h,ElectricError,'o')
%plot(log(h),log(ElectricError));

hold on


loglog(h,ElectricError)
%loglog(h,10^(1)*h.^(1.8))
hold on
%Pick a basis point for the triangle
xseed = 0.9*h(6)+h(5)*0.1;
%yseed = 0.0124;
yseed = 0.046;
%desiredSlope Of triangle
slope = 2;
%Another x point
xnext = h(4)*0.25+0.75*h(5);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'r')

text(xseed*0.7+0.3*xnext,ynext+0.008+0.0005,'1')
text(xseed-0.0011,0.5*yseed+0.5*ynext,'2')





title('Voronoi Tesselation')





%Now the ticks
%The x-axis
%leftaxis = 0.0077;
%rightaxis = 0.1+0.05;
%upaxis = 0.55;
%downaxis = 0.0048;

leftaxis  = 0.022;
rightaxis = 0.22;
upaxis = 1;
downaxis = 0.02;
axis([leftaxis rightaxis downaxis upaxis])
xlabel('Mesh Size')

ax = log10(leftaxis);
bx = log10(rightaxis);
tx = ax:(bx-ax)/10:bx;
tx = 10.^(tx);
ay = log10(downaxis);
by = log10(upaxis);
ty = ay:(by-ay)/10:by;
ty = 10.^(ty);

ylabel('Error On Electric Field')

%xticks(tx)
%yticks(ty)
grid on


ex = gca;
ex.FontSize =13;

%Plotting The magnetic Field

figure(2)
clf
loglog(h,MagneticError,'o')
hold on
loglog(h,MagneticError)

hold on

%Pick a basis point for the triangle
%xseed = h(4)*0.25+0.75*h(5);
%yseed = 0.45;
xseed = 0.9*h(6)+0.1*h(5);
yseed = 1.1;
%desiredSlope Of triangle
slope = 1;
%Another x point
%xnext = h(4);
xnext = h(5)*0.75+0.25*h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'r')


text(xseed*0.7+0.3*xnext,ynext+0.01+0.01+0.01+0.03,'1')
text(xseed-0.001/1.1,0.5*yseed+0.5*ynext,'1')










title('Voronoi Tesselation')
xlabel('Mesh Size')
ylabel('Error On Magnetic Field')


%axis([0.01-0.005 0.1+0.01 0.1+0.1 3])

%leftaxis = 0.0068*.99+0.01*0.119;
%rightaxis = 0.15;
%downaxis = 0.25;
%upaxis = 5;
leftaxis = 0.023;
rightaxis = 0.2;
downaxis = 0.8;
upaxis = 6.5;
axis([leftaxis rightaxis downaxis upaxis])
xlabel('Mesh Size')

ax = log10(leftaxis);
bx = log10(rightaxis);
tx = ax:(bx-ax)/10:bx;
tx = 10.^(tx);
ay = log10(downaxis);
by = log10(upaxis);
ty = ay:(by-ay)/10:by;
ty = 10.^(ty);

tx = [-2,-1,0];
tx = 10.^(tx);
ty = tx;
xtick=[0.1];
xticklab = cellstr(num2str(round(-log10(xtick(:))), '10^-^%d'));
set(gca,'XTick',xtick,'XTickLabel',xticklab,'TickLabelInterpreter','tex')

ytick=[1];
yticklab = cellstr(num2str(round(-log10(ytick(:))), '10^-^%d'));
set(gca,'YTick',ytick,'YTickLabel',yticklab,'TickLabelInterpreter','tex')



%XTick = [0.1,1];
%XTickLabels = cellstr(num2str(round(log10(XTick(:))), '10^%d'));
%xticks(tx)
%yticks(ty)
%H = axes;
%set(H, 'YTickLabels', {'10^{-1}'})
grid on

ex = gca;
ex.FontSize =13;
end
