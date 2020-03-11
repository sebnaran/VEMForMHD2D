%shape = 'quads';
%shape = 'triangles';
shape = 'voronoi';

switch shape
    case 'triangles'
%%%%%%%%%%%%%%%%%Triangles

h = [0.10101525445522107, 0.05018856132284956, 0.025141822757713456, 0.012525468249897755,...
     0.006261260829309998];
%at T = 10
%Relative Errors
%H1ElectricError   = [1.8264236664409103e-13, 1.2603008753614207e-13, 9.451755622655552e-14, 1.899965958302663e-14, 2.5771037369531976e-14];

%H1MagneticError   = [0.017587221510313302, 0.004736375161172553, 0.0014378566847645005, 0.0005305941758074996, 0.0002284356141592416];

%LSElectricError   = [1.5239963394052508e-13, 8.433919012013731e-14, 1.6921584480311475e-14, 2.933176784399617e-14, 4.601033333371091e-14];

%LSMagneticError   = [0.005103084317165196, 0.001857505654707116, 0.000839287240094878, 0.00043972530813441547, 0.00021749882299918895];

%PiecewiseElectric = [1.5745818304685657e-13, 1.0277316217015574e-13, 1.8219501456536164e-14, 2.6176284166876285e-14, 4.030409175422782e-14];

%PiecewiseMagnetic = [0.0037630061634813827, 0.0016210678916073037, 0.0008076051878031844, 0.00043579326056339933, 0.0002169325273292151];
%a T = 0.001
H1ElectricError   = [0.48874354214183974, 0.2324394772633422, 0.1741828387123998, 0.11787093529460302, 0.051087885163097];

H1MagneticError   = [0.00011483640545102537, 0.0001678616973624867, 0.000276681676024497, 0.0003875737928460475, 0.00032156229368384895];

LSElectricError   = [0.43449851065181944, 0.3752469987298846, 0.1955246052879086, 0.08112255760378784, 0.02559525996818712];

LSMagneticError   = [0.0005210643726138162, 0.0011020987987984505, 0.0010139074055721487, 0.0007384327153073039, 0.00041109126454675813];

PiecewiseElectric = [0.36861401892812384, 0.3494295416558225, 0.20445023170911208, 0.08863572640126705, 0.028008281350552833];

PiecewiseMagnetic = [0.00037747007156849053, 0.0008222163479924281, 0.0008880053771100764, 0.0007084827082902744, 0.0004062655726561863];


set(gca,'FontSize',15)
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


x = [xseed, xseed,xnext,ynext];
%secondtitle = 'Triangles';




%Plotting the electric field
figure(1)
clf
loglog(h,H1ElectricError,'o','LineWidth',3,'color','k');
hold on
loglog(h,LSElectricError,'d','LineWidth',3,'color','b');
loglog(h,PiecewiseElectric,'s','Linewidth',3,'color','r');
loglog(h,H1ElectricError,'LineWidth',2,'Color','k');
loglog(h,LSElectricError,'--','LineWidth',2,'Color','b');
loglog(h,PiecewiseElectric,':','Linewidth',2,'Color','r');

set(gca,'FontSize',15)
set(gca,'linewidth',2)
hold on
%Pick a basis point for the triangle
xseed = 0.9*h(5)+h(4)*0.1;
%yseed = 0.00005;
yseed = 0.06;
%desiredSlope Of triangle
slope = 2;
%Another x point
xnext = h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'LineWidth',2,'Color','k')

%text(xseed*0.7+0.3*xnext,ynext+0.003,'1','Fontsize',15)
%text(xseed-0.001,0.5*yseed+0.5*ynext,'2','Fontsize',15)





%title('Triangular Mesh')





%Now the ticks
%The x-axis
leftaxis = 0.005;
rightaxis = 0.13;
upaxis    = 0.6;
downaxis  = 0.01
%upaxis = 1e-12;
%downaxis = 1e-14;

axis([leftaxis rightaxis downaxis upaxis])

legend('E','LS', 'GI')
legend('Location','northwest')

%Plotting The magnetic Field
test = imread('TriangularMesh.png'); 
axes('position',[0.657 0.145 0.225 0.25]); 
imagesc(test)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
figure(2)
clf
loglog(h,H1MagneticError,'o','LineWidth',3,'color','k')
hold on
loglog(h,LSMagneticError,'d','LineWidth',3,'color','b')
loglog(h,PiecewiseMagnetic,'s','Linewidth',3,'color','r');
loglog(h,H1MagneticError,'LineWidth',2,'Color','k')
loglog(h,LSMagneticError,'--','LineWidth',2,'Color','b')
loglog(h,PiecewiseMagnetic,':','Linewidth',2,'Color','r');
set(gca,'FontSize',15)
set(gca,'linewidth',2)

hold on

%Pick a basis point for the triangle
xseed = h(4)*0.05+0.95*h(5);
%yseed = 0.0004;
yseed = 0.0006;
%desiredSlope Of triangle
slope = 1;
%Another x point
xnext = h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'LineWidth',2,'Color','k')


%text(xseed*0.7+0.3*xnext,ynext+0.01+0.01+0.01+0.01,'1','FontSize',15)
%text(xseed-0.0015/1.5,0.5*yseed+0.5*ynext,'1','FontSize',15)










%title('Triangular Mesh')
%xlabel('Mesh Size')
%ylabel('Error On Magnetic Field')


%axis([0.01-0.005 0.1+0.01 0.1+0.1 3])

leftaxis = 0.005;
rightaxis = 0.13;
downaxis = 0.0001;
upaxis = 0.02;


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

legend('E','LS','GI')
legend('Location','northwest')
%xticks(tx)
%yticks(ty)

%grid on
test = imread('TriangularMesh.png'); 
%axes('position',[0.657 0.145 0.225 0.25]); 
axes('position',[0.657 0.6 0.225 0.25]); 
imagesc(test)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);




























    case 'quads'
%%%%%%%%%%%%%%%%%%%PerturbedSquares
%h = [0.471405,0.235702,0.122975,0.0614875,0.0310816,0.0155408,0.00779781];
%The above was the older mesh-size

h = [0.16666666666666666, 0.08333333333333333, 0.043478260869565216, 0.021739130434782608,...
     0.010989010989010988];%, 0.005494505494505494,0.0027548209366391185];

%Relative Errors at T= 10
%H1ElectricError   = [2.7267586675653347e-14, 4.03314515737159e-14, 2.927438879556675e-14, 8.49980390217845e-15, 2.2959124731702042e-14];

%H1MagneticError   = [0.01558606216129967, 0.0041936058209108475, 0.0011853151440921195, 0.0003139996652133379, 9.892311480015528e-05];

%LSElectrciError   = [3.2672825947694215e-14, 2.048776422368143e-14, 8.725948753593509e-15, 1.6671539366358945e-14, 3.721414968979269e-14];

%LSMagneticError   = [0.007308257122218176, 0.0019222788124186676, 0.0005538252002841316, 0.00016236760701665836, 6.417161822471675e-05];

%PiecewiseElectric = [3.249455751265355e-14, 3.0593655720520954e-14, 7.898639885694773e-15, 2.5354706207923002e-14, 4.512693980079316e-14];

%PiecewiseMagnetic = [0.003745093493200769, 0.0010306922171819293, 0.0003221715233975197, 0.0001164921807566515, 5.721021737016075e-05];
%at T = 0.001
H1ElectricError   = [0.4001833324710447, 0.15730490544440484, 0.07325607117403106, 0.0363202085424494, 0.018686353487175508];

H1MagneticError   = [7.448553568903115e-05, 6.859588720819212e-05, 7.419458194388571e-05, 7.817367847723205e-05, 7.989026960269859e-05];

LSElectrciError   = [0.2216880313700157, 0.08105625680830125, 0.043273953508541005, 0.02582779100011497, 0.01350808314642758];

LSMagneticError   = [5.781334906835979e-05, 7.01240477168165e-05, 8.092060220251564e-05, 8.645198113891991e-05, 7.789667660187231e-05];

PiecewiseElectric = [0.12639604803263138, 0.0908528954727615, 0.051382561472738265, 0.022949058488054726, 0.008484068846564084];

PiecewiseMagnetic = [6.186554423130265e-05, 0.00015197079073428515, 0.00016778617231439185, 0.0001419596315359794, 9.962756816970764e-05];
figure(1)

clf

loglog(h,H1ElectricError,'o','LineWidth',3,'color','k')
hold on
loglog(h,LSElectricError,'d','LineWidth',3,'color','b')
loglog(h,PiecewiseElectric,'s','Linewidth',3,'color','r');
loglog(h,H1ElectricError,'LineWidth',2,'Color','k')
loglog(h,LSElectricError,'--','LineWidth',2,'Color','b')
loglog(h,PiecewiseElectric,':','Linewidth',2,'color','r');
%loglog(h,10^(1)*h.^(1.8))
hold on
%Pick a basis point for the triangle
xseed = 0.95*h(5)+0.05*h(4);
yseed = 0.000025;
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
set(gca,'linewidth',2)
%text(xseed*0.7+0.3*xnext,ynext+0.002,'1','FontSize',15)
%text(xseed-0.002,0.5*yseed+0.5*ynext,'2','FontSize',15)




%title('Perturbed Quadrilateral Mesh')
%axis([0.01 0.25 (0.001+0.0001)/2 .15])
%xlabel('Mesh Size')
%ylabel('Error On Electric Field')

leftaxis = 0.009;
rightaxis = 0.18;
%downaxis = 5e-15;
%upaxis = 2e-13;
downaxis = 0.005;
upaxis   = 0.6;
axis([leftaxis rightaxis downaxis upaxis])


ax = log10(leftaxis);
bx = log10(rightaxis);
tx = ax:(bx-ax)/10:bx;
tx = 10.^(tx);
ay = log10(downaxis);
by = log10(upaxis);
ty = ay:(by-ay)/10:by;
ty = 10.^(ty);

legend('E','LS', 'GI')
legend('Location','northwest')
%xti16cks(tx)
%yticks(ty)

test = imread('PerturbedSquaresMesh.png'); 
axes('position',[0.657 0.145 0.225 0.25]); 
imagesc(test)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);



%grid on



%Plotting The magnetic Field

figure(2)
clf
loglog(h,H1MagneticError,'o','LineWidth',3,'color','k')
hold on
loglog(h,LSMagneticError,'d','LineWidth',3,'color','b')
loglog(h,PiecewiseMagnetic,'s','Linewidth',3,'color','r');
loglog(h,H1MagneticError,'LineWidth',2,'Color','k')
loglog(h,LSMagneticError,'--','LineWidth',2,'Color','b')
loglog(h,PiecewiseMagnetic,':','Linewidth',2,'color','r');

hold on

%Pick a basis point for the triangle
xseed = h(4)*0.05+0.95*h(5);
%yseed = 0.0005;
yseed = 0.5;
%desiredSlope Of triangle
slope = 1;
%Another x point
xnext = h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'LineWidth',2,'Color','k')


%text(xseed*0.7+0.3*xnext,ynext+0.01+0.01,'1','FontSize',15)
%text(xseed-0.0015,0.5*yseed+0.5*ynext,'1','FontSize',15)

set(gca,'FontSize',15)
set(gca,'linewidth',2)







%title('Perturbed Quadrilateral Mesh')
%xlabel('Mesh Size')
%ylabel('Error On Magnetic Field')
%axis([0.01 0.25 0.75*10^(-1)+0.25*0.01 1])
%grid on

leftaxis = 0.009;
rightaxis = 0.18;
downaxis = 0.000025;
upaxis = 0.018;

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


legend('E','LS','GI')
legend('Location','northwest')
test = imread('PerturbedSquaresMesh.png'); 
%axes('position',[0.657 0.145 0.225 0.25]);
axes('position',[0.657 0.6 0.225 0.25]); 
imagesc(test)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
    case 'voronoi'
%%%%%%%%%%%%%%%%%%%%%%%%%%Voronoi

%h = [0.12803687993289598, 0.06772854614785964, 0.03450327796711771, 0.017476749542968805,...
%     0.008787156237382746];%, 0.004419676355414694,0.0022139558199118672];

%ElectricError = [0.4846671385368301, 0.17629056352803815, 0.0894535222510598, 0.013677866376455412,...
%    0.006616196310031363];

%MagneticError = [4.418879142501644, 2.4900590603987354, 1.2075527434363857, 0.5879063547722386,...
% 0.276556530752596];

h = [0.12803687993289598, 0.06772854614785964, 0.03450327796711771, 0.017476749542968805,0.008787156237382746];

%Relative Errors T = 10
%H1ElectricError   = [1.8797408072739968e-13, 1.8821346562922054e-13, 9.012194276752477e-05, 2.148529459704335e-05, 2.0511389455556706e-05];

%H1MagneticError   = [0.010779205273439429, 0.004237469191569036, 0.0014320154362075649, 0.0006728843023403707, 0.0003117634597221462];

%LSElectricError   = [1.6122733407768606e-13, 1.6041244994909393e-13, 0.0001854833326538939, 2.202155533671619e-05, 2.1664693085355392e-05];

%LSMagneticError   = [0.005066000784448221, 0.002120176619194025, 0.0010707824905052618, 0.0005635051458849925, 0.0002680255372191208];

%PiecewiseElectric = [1.1924594932743207e-13, 5.125210722136945e-08, 0.00022083459624554238, 9.854325999171794e-05, 4.0850541950503805e-05];

%PiecewiseMagnetic = [0.0028334185602056517, 0.0018302200399655019, 0.000889347760477469, 0.0004664211675678181, 0.00021958469607669622];

%T = 0.001
H1ElectricError   = [0.5645073666737741, 0.3573885511949236, 0.22068027586924627, 0.15402222503592145, 0.07607148840414828];

H1MagneticError   = [0.00039678106050693434, 0.0006153583067013272, 0.0007713827747413531, 0.0005530789787687338, 0.0004144089903016284];

LSElectricError   = [0.23980732007908334, 0.2759025481338045, 0.20258934570092071, 0.13549639978934805, 0.060316641752222136];

LSMagneticError   = [0.0002739624453643498, 0.0007756695424956423, 0.0008207291786216266, 0.0005935674833043064, 0.00040375463309435];

PiecewiseElectric = [0.3644447257582212, 0.3817105834690293, 0.19863967154758588, 0.08662247612455808, 0.02639833166289505];

PiecewiseMagnetic = [0.0007578055429542914, 0.001472562962940294, 0.0012334312974678992, 0.0007958185139008276, 0.0004184853291384077];
figure(1)

clf

loglog(h,H1ElectricError,'o','LineWidth',3,'color','k')
hold on
loglog(h,LSElectricError,'d','LineWidth',3,'color','b')
loglog(h,PiecewiseElectric,'s','Linewidth',3,'color','r');
loglog(h,H1ElectricError,'LineWidth',2,'Color','k')
loglog(h,LSElectricError,'--','LineWidth',2,'Color','b')
loglog(h,PiecewiseElectric,':','Linewidth',2,'color','r');
hold on
%Pick a basis point for the triangle
xseed = 0.9*h(5)+h(4)*0.1;
%yseed = 0.0124;
%yseed = 0.00015;
yseed = 0.01;
%desiredSlope Of triangle
slope = 2;
%Another x point
xnext = h(4)*0.95+0.05*h(5);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'LineWidth',2,'Color','k')

%text(xseed*0.7+0.3*xnext,ynext+0.001+0.0035,'1','FontSize',15)
%text(xseed-0.0012,0.5*yseed+0.5*ynext,'2','FontSize',15)
set(gca,'FontSize',15)
set(gca,'linewidth',2)



%title('Voronoi Tesselation')





%Now the ticks
%The x-axis
%leftaxis = 0.0077;
%rightaxis = 0.1+0.05;
%upaxis = 0.55;
%downaxis = 0.0048;

leftaxis  = 0.008;
rightaxis = 0.15;
%upaxis = 9e-4;
%downaxis = 6e-14;
upaxis = 0.6;
downaxis = 0.01;
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

legend('E','LS','GI')
legend('Location','northwest')


test = imread('VoronoiMesh.png'); 
%axes('position',[0.15 0.35 0.25 0.25]);
axes('position',[0.6 0.15 0.25 0.25]); 
imagesc(test)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);


%Plotting The magnetic Field

figure(2)
clf
loglog(h,H1MagneticError,'o','LineWidth',3,'color','k')
hold on
loglog(h,LSMagneticError,'d','LineWidth',3,'color','b')
loglog(h,PiecewiseMagnetic,'s','Linewidth',3,'color','r');
loglog(h,H1MagneticError,'LineWidth',2,'Color','k')
loglog(h,LSMagneticError,'--','LineWidth',2,'Color','b')
loglog(h,PiecewiseMagnetic,':','Linewidth',2,'color','r');


hold on

%Pick a basis point for the triangle
%xseed = h(4)*0.25+0.75*h(5);
%yseed = 0.45;
xseed = 0.95*h(5)+0.05*h(4);
yseed = 0.0015;
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


%text(xseed*0.7+0.3*xnext,ynext+0.01+0.01+0.01+0.01+0.01,'1','FontSize',15)
%text(xseed-0.0015/1.5,0.5*yseed+0.5*ynext,'1','fontSize',15)




set(gca,'FontSize',15)
set(gca,'linewidth',2)




%title('Voronoi Tesselation')
%xlabel('Mesh Size')
%ylabel('Error On Magnetic Field')


%axis([0.01-0.005 0.1+0.01 0.1+0.1 3])

%leftaxis = 0.0068*.99+0.01*0.119;
%rightaxis = 0.15;
%downaxis = 0.25;
%upaxis = 5;
leftaxis = 0.008;
rightaxis = 0.15;
downaxis = 0.0001;
upaxis = 0.04;
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
legend('E','LS','GI')
legend('Location','northwest')

test = imread('VoronoiMesh.png'); 
%axes('position',[0.657 0.145 0.225 0.25]);
axes('position',[0.657 0.6 0.225 0.25]); 
imagesc(test)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
end