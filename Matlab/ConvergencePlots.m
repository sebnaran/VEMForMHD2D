%shape = 'quads';
%shape = 'triangles';
shape = 'voronoi';

switch shape
    case 'triangles'
%%%%%%%%%%%%%%%%%Triangles

h = [0.10101525445522107, 0.05018856132284956, 0.025141822757713456, 0.012525468249897755,...
     0.006261260829309998];

%H1ElectricError =  [0.5098721095770006, 0.12175648676696989, 0.03224235083294914, 0.007922421692486309, 0.001949904177034469];

%H1MagneticError = [2.0042181007879325, 0.9781866298047875, 0.5502616339498934, 0.28069706397737304, 0.13768142439140277];

%LSElectricError = [0.14818729871298789, 0.04458205952586334, 0.01236690803569729, 0.003386789150707841, 0.0008478943600952674];

%LSMagneticError = [1.737093523457867, 0.8830857310478574, 0.5308519199985781, 0.2801388010942833, 0.1374005242633508];

%PiecewiseElectric = [0.15617332183125077, 0.04720367855318061, 0.013734498731550082, 0.003706965600792301, 0.0009266598253482083];

%PiecewiseMagnetic = [1.7257613098011293, 0.8821623227726677, 0.5308026621064391, 0.2800706220693727, 0.13739725521090834];
%New trial.
%PiecewiseElectric = [0.15565380780196314, 0.06619434712739114, 0.016493843607543232, 0.004374138191195906, 0.0010836239662052856];
%PiecewiseMagnetic = [1.5431901128336851, 0.9569453749129787, 0.541891998918303, 0.28335839603304325, 0.14036938458439813];
%secondtitle = 'Triangles';
%Using the centroid
%PiecewiseElectric = [0.15617332183123447, 0.047203678553183685, 0.013734498731569515, 0.003706965600802786, 0.0009266598253584832];
%PiecewiseMagnetic = [1.632657414334451, 0.8685300411286138, 0.5292523226138253, 0.2798912912511224, 0.13737942008748644];

%Relative Errors
H1ElectricError   = [0.0066018410182922815, 0.0016411111026913435, 0.00044293381731688947, 0.00010903206288271363, 2.69209350699779e-05];

H1MagneticError   = [0.013916958988976465, 0.006632633060020177, 0.00372081132305619, 0.001893495679319807, 0.0009284458222196577];

LSElectricError   = H1ElectricError;

LSMagneticError   = H1MagneticError;

PiecewiseElectric = [0.002142036493849704, 0.0006506890054289756, 0.0001895306417710418, 5.11705921121986e-05, 1.2792573358683945e-05];

PiecewiseMagnetic = [0.011056580633680776, 0.0058626406802457474, 0.0035694745399296127, 0.001887214331824036, 0.0009262413268502271];
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
loglog(h,LSElectricError,'LineWidth',2,'Color','b');
loglog(h,PiecewiseElectric,'Linewidth',2,'Color','r');

set(gca,'FontSize',15)
set(gca,'linewidth',2)
hold on
%Pick a basis point for the triangle
xseed = 0.9*h(5)+h(4)*0.1;
yseed = 0.00005;
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
upaxis = 0.012;
downaxis = 10e-6;

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
loglog(h,LSMagneticError,'LineWidth',2,'Color','b')
loglog(h,PiecewiseMagnetic,'Linewidth',2,'Color','r');
set(gca,'FontSize',15)
set(gca,'linewidth',2)

hold on

%Pick a basis point for the triangle
xseed = h(4)*0.05+0.95*h(5);
yseed = 0.0015;
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
downaxis = 0.0008;
upaxis = 0.015;

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
axes('position',[0.657 0.145 0.225 0.25]); 
imagesc(test)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);




























    case 'quads'
%%%%%%%%%%%%%%%%%%%PerturbedSquares
%h = [0.471405,0.235702,0.122975,0.0614875,0.0310816,0.0155408,0.00779781];
%The above was the older mesh-size

h = [0.16666666666666666, 0.08333333333333333, 0.043478260869565216, 0.021739130434782608,...
     0.010989010989010988];%, 0.005494505494505494,0.0027548209366391185];

%H1ElectricError = [0.1505488392083424, 0.07540240822978826, 0.01964292814740662, 0.00461064875933448, 0.0010155804421071366];

%H1MagneticError = [1.3485966107898497, 0.515270877390314, 0.22681651972361622, 0.10557908129467171, 0.05201601043760578];
                                                      
%LSElectricError = [0.10257125554190669, 0.04721190787824579, 0.014863112248450432, 0.0036221150547460566, 0.0007750246467491926];
%LSMagneticError = [0.9331853803694574, 0.45839095006777036, 0.2045238649645601, 0.10048477597864197, 0.05047566104015773];

%PiecewiseElectric = [0.06401026321938064, 0.023402075732633493, 0.010531654108504205, 0.0026036967096083725, 0.0004036108290974887];
%PiecewiseMagnetic = [0.9047620434005373, 0.44128757090293436, 0.20226301729409893, 0.10020865068781082, 0.05040840568641607];
%New trial
%PiecewiseElectric = [0.0512350094121807, 0.023635417304269993, 0.007083805155839038, 0.002505228754786083, 0.0017995519817493328];
%PiecewiseMagnetic = [0.8046433434267048, 0.4318927166148368, 0.20971710032225624, 0.10885451829483941, 0.05581372009656225];
%secondtitle = 'On Perturbed squares';
%Using the centroid
%PiecewiseElectric = [0.06399634115666948, 0.023420083928144014, 0.010532075496569855, 0.0026037656684741953, 0.00040364092518334354];
%PiecewiseMagnetic = [0.5496670990445647, 0.4000805616528737, 0.1985465027804371, 0.09980004534231372, 0.0502907138579932];
%Plotting the electric field

%Relative Errors
H1ElectricError   = [0.001774417930875261, 0.0010062351690206253, 0.0002637642679525955, 6.217070950806123e-05, 1.3907865683591227e-05];

H1MagneticError   = [0.009534427732761534, 0.0035522498596006944, 0.001532968126216255, 0.0007127225108885281, 0.00035083375879386953];

LSElectricError   = H1ElectricError;

LSMagneticError   = H1MagneticError;

PiecewiseElectric = [0.0008751450878900018, 0.0003225251494994439, 0.00014529896989628577, 3.593982412134691e-05, 5.572181166640876e-06];

PiecewiseMagnetic = [0.003722607642478202, 0.002700378986463313, 0.0013390200102385234, 0.0006729085824221313, 0.00033906931857913345];

figure(1)

clf

loglog(h,H1ElectricError,'o','LineWidth',3,'color','k')
hold on
loglog(h,LSElectricError,'d','LineWidth',3,'color','b')
loglog(h,PiecewiseElectric,'s','Linewidth',3,'color','r');
loglog(h,H1ElectricError,'LineWidth',2,'Color','k')
loglog(h,LSElectricError,'LineWidth',2,'Color','b')
loglog(h,PiecewiseElectric,'Linewidth',2,'color','r');
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
downaxis = 0.00004;
upaxis = 0.0022;

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
loglog(h,LSMagneticError,'LineWidth',2,'Color','b')
loglog(h,PiecewiseMagnetic,'Linewidth',2,'color','r');

hold on

%Pick a basis point for the triangle
xseed = h(4)*0.05+0.95*h(5);
yseed = 0.0005;
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
downaxis = 0.00025;
upaxis = 0.012;

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
axes('position',[0.657 0.145 0.225 0.25]); 
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

%H1ElectricError = [0.6402171603657321, 0.2191316614798317, 0.05161479281147327, 0.013327689044599616, 0.003435073380290061];

%H1MagneticError = [3.3226284919819693, 1.6827177392771193, 0.7955836030979955, 0.3792426183789603, 0.18192650249011413];

%LSElectricError = [0.3682110522050534, 0.15810796882225037, 0.03953170336385856, 0.010469264282408906, 0.003850058673999911];

%LSMagneticError = [2.697860282739323, 1.504084831376333, 0.74567700206153, 0.36236339153342956, 0.17296289875225518];
%Original (using sum x_i/N as the point)
%PiecewiseElectric = [0.2844938672969478, 0.0957880521111463, 0.08430848536317731, 0.010034323386856954, 0.006058071946101146];
%PiecewiseMagnetic = [2.526393278930101, 1.4568176720822381, 0.7152759426644181, 0.35076653688237175, 0.1663073302200001];
%New
%PiecewiseElectric = [0.2742857252259966, 0.09915374034956144, 0.08463413832732498, 0.01187454501946442, 0.006399316962147336];
%PiecewiseMagnetic = [2.574014260656867, 1.495773132863638, 0.7397827741185969, 0.366456406959799, 0.1755798605185168];
%secondtitle='Voronoi';
%Using the centroid
%PiecewiseElectric = [0.27746878534448505, 0.09546265049240821, 0.08390558807034822, 0.009715678669095975, 0.006066470576797099];
%PiecewiseMagnetic = [2.534962998626922, 1.449425945308744, 0.7070098149587313, 0.34659967018971993, 0.16376959355435852];

%Relative Errors
H1ElectricError   = [0.00845940149671967, 0.0029779545406830566, 0.0007088809050338395, 0.00018380482499734087, 4.7407445420933354e-05];

H1MagneticError   = [0.02310924244551625, 0.011438436350391148, 0.005374986983259667, 0.0025580663010301456, 0.0012266852950447824];

LSElectricError   = H1ElectricError;

LSMagneticError   = H1MagneticError;

PiecewiseElectric = [0.003814593166903991, 0.001316450788757688, 0.0011579456279979844, 0.00013411656251855646, 8.374815634413833e-05];

PiecewiseMagnetic = [0.01716968221857468, 0.00978890083640975, 0.004768700680518675, 0.002337061575947633, 0.0011041751607145534];


figure(1)

clf

loglog(h,H1ElectricError,'o','LineWidth',3,'color','k')
hold on
loglog(h,LSElectricError,'d','LineWidth',3,'color','b')
loglog(h,PiecewiseElectric,'s','Linewidth',3,'color','r');
loglog(h,H1ElectricError,'LineWidth',2,'Color','k')
loglog(h,LSElectricError,'LineWidth',2,'Color','b')
loglog(h,PiecewiseElectric,'Linewidth',2,'color','r');
hold on
%Pick a basis point for the triangle
xseed = 0.9*h(5)+h(4)*0.1;
%yseed = 0.0124;
yseed = 0.00015;
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
upaxis = 0.01;
downaxis = 0.000025;
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
axes('position',[0.657 0.145 0.225 0.25]); 
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
loglog(h,LSMagneticError,'LineWidth',2,'Color','b')
loglog(h,PiecewiseMagnetic,'Linewidth',2,'color','r');


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
downaxis = 0.0008;
upaxis = 0.026;
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
axes('position',[0.657 0.145 0.225 0.25]); 
imagesc(test)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
end