HEcoorx
HEcoory
HE

HBcoorx
HBcoory
HBx
HBy

figure(1)
clf
x = Bcoorx' ; y = Bcoory' ; z = Bx' ;
dt = delaunayTriangulation(x,y) ;
tri = dt.ConnectivityList ;
xi = dt.Points(:,1) ; 
yi = dt.Points(:,2) ; 
F = scatteredInterpolant(x,y,z);
zi = F(xi,yi) ;
trisurf(tri,xi,yi,zi) 
view(2)
%view(90,0)
shading interp
set(gca,'FontSize',15)
%Set(gca,'linewidth',2)
%hold on
%plot ( [-1,-1],[1,-1],'LineWidth',4,'color','k')
%plot ( [-1,1],[-1,-1],'LineWidth',4,'color','k')
%plot ( [1,1],[-1,1],'LineWidth',4,'color','k')        
%plot ( [1,-1],[1,1],'LineWidth',4,'color','k')  

%Sideways
%[x y] = meshgrid(-1:0.1:1);
%z = -0.15*ones(size(x, 1));
%surf(x,y,z)
%set(gca,'Color','k')
%plot([-1,0,-0.15],[1,0,-0.15],'LineWidth',4,'color','k')
%Bxex = zeros(length(Bx),1);
%Byex = zeros(length(By),1);
%for i=1:length(Bcoorx)
%    [Bxex(i),Byex(i)] = Bfield(Bcoorx(i),Bcoory(i));
%end
%hold on
%figure(2)
%x = Bcoorx' ; y = Bcoory' ; z = Bxex ;
%dt = delaunayTriangulation(x,y) ;
%tri = dt.ConnectivityList ;
%xi = dt.Points(:,1) ; 
%yi = dt.Points(:,2) ; 
%F = scatteredInterpolant(x,y,z);
%zi = F(xi,yi) ;
%trisurf(tri,xi,yi,zi) 
%view(2)
%view(90,0)
%shading interp
figure(2)
y = [-1:0.001:1];
[Bxexact,trash] = Bfield(1,y);
plot(y,Bxexact,'Linewidth',2,'Color','k')
set(gca,'FontSize',15)
set(gca,'linewidth',2)
    
 