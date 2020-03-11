MFDQuadSimulations
MFDTrigSimulations
MFDVoronoiSimulations

figure(1)
clf
plot(trigtime,trigdiv,'Linewidth',2,'color','k')
hold on
plot(quadtime,quaddiv,'--','Linewidth',2,'color','b')
plot(voronoitime,voronoidiv,':','Linewidth',2,'color','r')
legend('T','Q', 'V')
legend('Location','east')
set(gca,'FontSize',15)
set(gca,'linewidth',2)
%Export with size width = 15cms length = 12cms
%test = imread('TriangularMesh.png'); 
%axes('position',[0.657 0.35 0.225 0.25]); 
%imagesc(test)
%set(gca,'YTickLabel',[]);
%set(gca,'XTickLabel',[]);