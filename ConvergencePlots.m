shape = 'quads';
%shape = 'triangles';
%shape = 'voronoi';

switch shape
    case 'triangles'
%%%%%%%%%%%%%%%%%Triangles

h = [0.10101525445522107, 0.05018856132284956, 0.025141822757713456, 0.012525468249897755,...
     0.006261260829309998];

ElectricError =  [0.5098721095770006, 0.12175648676696989, 0.03224235083294914, 0.007922421692486309, 0.001949904177034469];

MagneticError = [2.0042181007879325, 0.9781866298047875, 0.5502616339498934, 0.28069706397737304, 0.13768142439140277];

%secondtitle = 'Triangles';




%Plotting the electric field

figure(1)

clf

loglog(h,ElectricError,'o','LineWidth',3)
%plot(log(h),log(ElectricError));

hold on


loglog(h,ElectricError,'LineWidth',2,'Color','k')
set(gca,'FontSize',15)
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

loglog(x,y,'LineWidth',2,'Color','k')

text(xseed*0.7+0.3*xnext,ynext+0.003,'1','Fontsize',15)
text(xseed-0.001,0.5*yseed+0.5*ynext,'2','Fontsize',15)





%title('Triangular Mesh')





%Now the ticks
%The x-axis
leftaxis = 0.005;
rightaxis = 0.13;
upaxis = 0.65;
downaxis = 0.0015;

axis([leftaxis rightaxis downaxis upaxis])
%xlabel('Mesh Size')

ax = log10(leftaxis);
bx = log10(rightaxis);
tx = ax:(bx-ax)/10:bx;
tx = 10.^(tx);
ay = log10(downaxis);
by = log10(upaxis);
ty = ay:(by-ay)/10:by;
ty = 10.^(ty);

%ylabel('Error On Electric Field')

%xticks(tx)
%yticks(ty)
%grid on




%Plotting The magnetic Field

figure(2)
clf
loglog(h,MagneticError,'o','LineWidth',3)
hold on
loglog(h,MagneticError,'LineWidth',2,'Color','k')
set(gca,'FontSize',15)


hold on

%Pick a basis point for the triangle
xseed = h(4)*0.05+0.95*h(5);
yseed = 0.18;
%desiredSlope Of triangle
slope = 1;
%Another x point
xnext = h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'LineWidth',2,'Color','k')


text(xseed*0.7+0.3*xnext,ynext+0.01+0.01+0.01+0.01,'1','FontSize',15)
text(xseed-0.0015/1.5,0.5*yseed+0.5*ynext,'1','FontSize',15)










%title('Triangular Mesh')
%xlabel('Mesh Size')
%ylabel('Error On Magnetic Field')


%axis([0.01-0.005 0.1+0.01 0.1+0.1 3])

leftaxis = 0.005;
rightaxis = 0.13;
downaxis = 0.125;
upaxis = 2.15;

axis([leftaxis rightaxis downaxis upaxis])
%xlabel('Mesh Size')

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

%grid on





























    case 'quads'
%%%%%%%%%%%%%%%%%%%PerturbedSquares
%h = [0.471405,0.235702,0.122975,0.0614875,0.0310816,0.0155408,0.00779781];
%The above was the older mesh-size

h = [0.16666666666666666, 0.08333333333333333, 0.043478260869565216, 0.021739130434782608,...
     0.010989010989010988];%, 0.005494505494505494,0.0027548209366391185];

ElectricError = [0.1505488392083424, 0.07540240822978826, 0.01964292814740662, 0.00461064875933448, 0.0010155804421071366];

MagneticError = [1.3485966107898497, 0.515270877390314, 0.22681651972361622, 0.10557908129467171, 0.05201601043760578];
                                                      

%secondtitle = 'On Perturbed squares';


%Plotting the electric field

figure(1)

clf

loglog(h,ElectricError,'o','LineWidth',3)
%plot(log(h),log(ElectricError));

hold on


loglog(h,ElectricError,'LineWidth',2,'Color','k')
%loglog(h,10^(1)*h.^(1.8))
hold on
%Pick a basis point for the triangle
xseed = 0.95*h(5)+0.05*h(4);
yseed = 0.0016;
%desiredSlope Of triangle
slope = 2;
%Another x point
xnext = h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'r','LineWidth',2,'Color','k')
set(gca,'FontSize',15)
text(xseed*0.7+0.3*xnext,ynext+0.002,'1','FontSize',15)
text(xseed-0.002,0.5*yseed+0.5*ynext,'2','FontSize',15)





%title('Perturbed Quadrilateral Mesh')
%axis([0.01 0.25 (0.001+0.0001)/2 .15])
%xlabel('Mesh Size')
%ylabel('Error On Electric Field')

leftaxis = 0.009;
rightaxis = 0.18;
downaxis = 0.00091;
upaxis = 0.16;

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





%grid on



%Plotting The magnetic Field

figure(2)
clf
loglog(h,MagneticError,'o','LineWidth',3)
hold on
loglog(h,MagneticError,'LineWidth',2,'Color','k')



hold on

%Pick a basis point for the triangle
xseed = h(4)*0.05+0.95*h(5);
yseed = 0.065;
%desiredSlope Of triangle
slope = 1;
%Another x point
xnext = h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'LineWidth',2,'Color','k')


text(xseed*0.7+0.3*xnext,ynext+0.01+0.01,'1','FontSize',15)
text(xseed-0.0015,0.5*yseed+0.5*ynext,'1','FontSize',15)

set(gca,'FontSize',15)








%title('Perturbed Quadrilateral Mesh')
%xlabel('Mesh Size')
%ylabel('Error On Magnetic Field')
%axis([0.01 0.25 0.75*10^(-1)+0.25*0.01 1])
%grid on

leftaxis = 0.009;
rightaxis = 0.18;
downaxis = 0.047;
upaxis = 1.5;

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







    case 'voronoi'
%%%%%%%%%%%%%%%%%%%%%%%%%%Voronoi

%h = [0.12803687993289598, 0.06772854614785964, 0.03450327796711771, 0.017476749542968805,...
%     0.008787156237382746];%, 0.004419676355414694,0.0022139558199118672];

%ElectricError = [0.4846671385368301, 0.17629056352803815, 0.0894535222510598, 0.013677866376455412,...
%    0.006616196310031363];

%MagneticError = [4.418879142501644, 2.4900590603987354, 1.2075527434363857, 0.5879063547722386,...
% 0.276556530752596];

h = [0.12803687993289598, 0.06772854614785964, 0.03450327796711771, 0.017476749542968805,0.008787156237382746];

ElectricError = [0.6402171603657321, 0.2191316614798317, 0.05161479281147327, 0.013327689044599616, 0.003435073380290061];

MagneticError = [3.3226284919819693, 1.6827177392771193, 0.7955836030979955, 0.3792426183789603, 0.18192650249011413];




%secondtitle='Voronoi';



figure(1)

clf

loglog(h,ElectricError,'o','LineWidth',3)
%plot(log(h),log(ElectricError));

hold on


loglog(h,ElectricError,'LineWidth',2,'Color','k')
%loglog(h,10^(1)*h.^(1.8))
hold on
%Pick a basis point for the triangle
xseed = 0.9*h(5)+h(4)*0.1;
%yseed = 0.0124;
yseed = 0.0055;
%desiredSlope Of triangle
slope = 2;
%Another x point
xnext = h(4)*0.95+0.05*h(5);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'LineWidth',2,'Color','k')

text(xseed*0.7+0.3*xnext,ynext+0.001+0.0035,'1','FontSize',15)
text(xseed-0.0012,0.5*yseed+0.5*ynext,'2','FontSize',15)
set(gca,'FontSize',15)




%title('Voronoi Tesselation')





%Now the ticks
%The x-axis
%leftaxis = 0.0077;
%rightaxis = 0.1+0.05;
%upaxis = 0.55;
%downaxis = 0.0048;

leftaxis  = 0.007;
rightaxis = 0.15;
upaxis = 0.8;
downaxis = 0.002;
axis([leftaxis rightaxis downaxis upaxis])
%xlabel('Mesh Size')

ax = log10(leftaxis);
bx = log10(rightaxis);
tx = ax:(bx-ax)/10:bx;
tx = 10.^(tx);
ay = log10(downaxis);
by = log10(upaxis);
ty = ay:(by-ay)/10:by;
ty = 10.^(ty);

%ylabel('Error On Electric Field')

%xticks(tx)
%yticks(ty)
%grid on




%Plotting The magnetic Field

figure(2)
clf
loglog(h,MagneticError,'o','LineWidth',3)
hold on
loglog(h,MagneticError,'LineWidth',2,'Color','k')



hold on

%Pick a basis point for the triangle
%xseed = h(4)*0.25+0.75*h(5);
%yseed = 0.45;
xseed = 0.95*h(5)+0.05*h(4);
yseed = 0.22;
%desiredSlope Of triangle
slope = 1;
%Another x point
%xnext = h(4);
xnext = h(5)*0.05+0.95*h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'LineWidth',2,'Color','k')


text(xseed*0.7+0.3*xnext,ynext+0.01+0.01+0.01+0.01+0.01,'1','FontSize',15)
text(xseed-0.0015/1.5,0.5*yseed+0.5*ynext,'1','fontSize',15)




set(gca,'FontSize',15)





%title('Voronoi Tesselation')
%xlabel('Mesh Size')
%ylabel('Error On Magnetic Field')


%axis([0.01-0.005 0.1+0.01 0.1+0.1 3])

%leftaxis = 0.0068*.99+0.01*0.119;
%rightaxis = 0.15;
%downaxis = 0.25;
%upaxis = 5;
leftaxis = 0.007;
rightaxis = 0.15;
downaxis = 0.16;
upaxis = 3.6;
axis([leftaxis rightaxis downaxis upaxis])
%xlabel('Mesh Size')

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

%grid on























end

















