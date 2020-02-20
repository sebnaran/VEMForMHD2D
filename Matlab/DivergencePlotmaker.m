clear all

H1QuadSimulations
H1TrigSimulations
H1VoronoiSimulations

H1quaddiv    = quaddiv;
H1trigdiv    = trigdiv;
H1voronoidiv = voronoidiv;
clear quaddiv trigdiv voronoidiv;

LSQuadSimulations
LSTrigSimulations
LSVoronoiSimulations

LSquaddiv    = quaddiv;
LStrigdiv    = trigdiv;
LSvoronoidiv = voronoidiv;
clear quaddiv trigdiv voronoidiv;

PWQuadSimulations
PWTrigSimulations
PWVoronoiSimulations

PWquaddiv    = quaddiv;
PWtrigdiv    = trigdiv;
PWvoronoidiv = voronoidiv;
clear quaddiv trigdiv voronoidiv;

figure(1)
clf
plot(trigtime,H1trigdiv,'Linewidth',2,'color','k')
hold on
plot(trigtime,LStrigdiv,'Linewidth',2,'color','b')
plot(trigtime,PWtrigdiv,'Linewidth',2,'color','r')
legend('E','LS', 'GI')
legend('Location','southeast')
set(gca,'FontSize',15)
set(gca,'linewidth',2)
%Export with size width = 15cms length = 12cms
test = imread('TriangularMesh.png'); 
axes('position',[0.657 0.35 0.225 0.25]); 
imagesc(test)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

figure(2)
clf
plot(quadtime,H1quaddiv,'Linewidth',2,'color','k')
hold on
plot(quadtime,LSquaddiv,'Linewidth',2,'color','b')
plot(quadtime,PWquaddiv,'Linewidth',2,'color','r')
legend('E','LS', 'GI')
legend('Location','southeast')
set(gca,'FontSize',15)
set(gca,'linewidth',2)
test = imread('PerturbedSquaresMesh.png'); 
axes('position',[0.657 0.35 0.225 0.25]); 
imagesc(test)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

figure(3)
clf
plot(voronoitime,H1voronoidiv,'Linewidth',2,'color','k')
hold on
plot(voronoitime,LSvoronoidiv,'Linewidth',2,'color','b')
plot(voronoitime,PWvoronoidiv,'Linewidth',2,'color','r')
legend('E','LS', 'GI')
legend('Location','southeast')
set(gca,'FontSize',15)
set(gca,'linewidth',2)
test = imread('VoronoiMesh.png'); 
axes('position',[0.657 0.35 0.225 0.25]); 
imagesc(test)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
