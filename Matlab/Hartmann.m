HEcoorx
HEcoory
HE

HBcoorx
HBcoory
HBx
HBy

figure(1)
x = Bcoorx' ; y = Bcoory' ; z = Bx' ;
dt = delaunayTriangulation(x,y) ;
tri = dt.ConnectivityList ;
xi = dt.Points(:,1) ; 
yi = dt.Points(:,2) ; 
F = scatteredInterpolant(x,y,z);
zi = F(xi,yi) ;
trisurf(tri,xi,yi,zi) 
view(2)
shading interp

Bxex = zeros(length(Bx),1);
Byex = zeros(length(By),1);
for i=1:length(Bcoorx)
    [Bxex(i),Byex(i)] = Bfield(Bcoorx(i),Bcoory(i));
end

figure(2)
x = Bcoorx' ; y = Bcoory' ; z = Bxex ;
dt = delaunayTriangulation(x,y) ;
tri = dt.ConnectivityList ;
xi = dt.Points(:,1) ; 
yi = dt.Points(:,2) ; 
F = scatteredInterpolant(x,y,z);
zi = F(xi,yi) ;
trisurf(tri,xi,yi,zi) 
view(2)
shading interp
    
    
 