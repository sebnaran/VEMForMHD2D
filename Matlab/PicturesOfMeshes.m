  
pgon = polyshape([-1 -1 1 1],[1 -1 -1 1]);
tr = triangulation(pgon);
model = createpde;
tnodes = tr.Points';
telements = tr.ConnectivityList';

geometryFromMesh(model,tnodes,telements);
pdegplot(model)

generateMesh(model,'Hmax',0.3); 
figure(1)
Trigs = pdemesh(model,'edgecolor','black');
%Trigs = pdegplot(model);
set(Trigs,'linewidth',4)
set(Trigs,'MarkerEdgeColor',[0,0,0])
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);       
leftaxis  = -1;
rightaxis = 1;
upaxis    = 1;
downaxis  = -1;

axis([leftaxis rightaxis downaxis upaxis])
hold on
plot ( [-1,-1],[1,-1],'LineWidth',4,'color','k')
plot ( [-1,1],[-1,-1],'LineWidth',4,'color','k')        
plot ( [1,1],[-1,1],'LineWidth',4,'color','k')        
plot ( [1,-1],[1,1],'LineWidth',4,'color','k')      
pbaspect([1 1 1])
     

figure(2)
clf
x = 2*rand(50,1)-1;
y = 2*rand(50,1)-1;
Vors = voronoi(x,y);[
set(Vors,'linewidth',4)
set(Vors,'MarkerEdgeColor',[0,0,0])
set(Vors,'Color','k')
hold on
plot ( [-1,-1],[1,-1],'LineWidth',4,'color','k')
plot ( [-1,1],[-1,-1],'LineWidth',4,'color','k')        
plot ( [1,1],[-1,1],'LineWidth',4,'color','k')        
plot ( [1,-1],[1,1],'LineWidth',4,'color','k')      
pbaspect([1 1 1])
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]); 
pbaspect([1 1 1])
   
