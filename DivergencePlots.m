shape = 'quads';
%shape = 'triangles';
%shape = 'voronoi';

switch shape
    case 'triangles'
%%%%%%%%%%%%%%%%%Triangles

h = [0.10101525445522107, 0.05018856132284956, 0.025141822757713456, 0.012525468249897755,...
     0.006261260829309998];

TrigDivErr =  [3.002976829137562, 2.1064860592176657, 1.0410036970349146, 0.5270560502969557, 0.2600873697255153];

%Plotting the electric field

figure(1)

clf

loglog(h,TrigDivErr,'o','LineWidth',3)
%plot(log(h),log(ElectricError));

hold on


loglog(h,TrigDivErr,'LineWidth',2,'Color','k')
set(gca,'FontSize',15)
%loglog(h,10^(1)*h.^(1.8))
hold on
%Pick a basis point for the triangle
xseed = 0.9*h(5)+h(4)*0.1;
yseed = 0.4;
%desiredSlope Of triangle
slope = 1;
%Another x point
xnext = h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'LineWidth',2,'Color','k')

text(xseed*0.7+0.3*xnext,ynext+0.09,'1','Fontsize',15)
text(xseed-0.001,0.5*yseed+0.5*ynext,'1','Fontsize',15)





%title('Triangular Mesh')





%Now the ticks
%The x-axis
leftaxis = 0.005;
rightaxis = 0.13;
upaxis = 3.2;
downaxis = 0.23;

axis([leftaxis rightaxis downaxis upaxis])


ax = log10(leftaxis);
bx = log10(rightaxis);
tx = ax:(bx-ax)/10:bx;
tx = 10.^(tx);
ay = log10(downaxis);
by = log10(upaxis);
ty = ay:(by-ay)/10:by;
ty = 10.^(ty);




    case 'quads'
%%%%%%%%%%%%%%%%%%%PerturbedSquares
%h = [0.471405,0.235702,0.122975,0.0614875,0.0310816,0.0155408,0.00779781];
%The above was the older mesh-size

h = [0.16666666666666666, 0.08333333333333333, 0.043478260869565216, 0.021739130434782608,...
     0.010989010989010988];%, 0.005494505494505494,0.0027548209366391185];

QuadDivErr = [0.8690333685771567, 0.525626980861213, 0.28560170887820774, 0.14364975671067606, 0.07430754098528912];                                                

figure(1)
clf
loglog(h,QuadDivErr,'o','LineWidth',3)
hold on
loglog(h,QuadDivErr,'LineWidth',2,'Color','k')



hold on

%Pick a basis point for the triangle
xseed = h(4)*0.05+0.95*h(5);
yseed = 0.09;
%desiredSlope Of triangle
slope = 1;
%Another x point
xnext = h(4);


b = log10(yseed/(xseed^slope));


ynext = 10^(b)*xnext^slope;


x = [xseed, xseed, xnext, xseed];
y = [yseed, ynext, ynext, yseed];

loglog(x,y,'LineWidth',2,'Color','k')


text(xseed*0.7+0.3*xnext,ynext+0.02,'1','FontSize',15)
text(xseed-0.0015,0.5*yseed+0.5*ynext,'1','FontSize',15)

set(gca,'FontSize',15)

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


    case 'voronoi'
%%%%%%%%%%%%%%%%%%%%%%%%%%Voronoi

h = [0.12803687993289598, 0.06772854614785964, 0.03450327796711771, 0.017476749542968805,0.008787156237382746];

VoronoiDivErr = [1.5986694766999625, 0.8641749392831602, 0.35710840611522443, 0.15787684428759713, 0.07029059959994698];


figure(1)
clf
loglog(h,VoronoiDivErr,'o','LineWidth',3)
hold on
loglog(h,VoronoiDivErr,'LineWidth',2,'Color','k')



hold on

%Pick a basis point for the triangle
%xseed = h(4)*0.25+0.75*h(5);
%yseed = 0.45;
xseed = 0.95*h(5)+0.05*h(4);
yseed = 0.11;
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


text(xseed*0.7+0.3*xnext,ynext+0.01+0.01+0.01,'1','FontSize',15)
text(xseed-0.0015/1.5,0.5*yseed+0.5*ynext,'1','fontSize',15)

set(gca,'FontSize',15)

leftaxis = 0.007;
rightaxis = 0.15;
downaxis = 0.06;
upaxis = 1.7;
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
end