clear all

fQhalf
fQTF
fQone

fFQhalf
fFQone
fFQTF

figure(1)
clf
plot(Qhalf,FQhalf,'--','Linewidth',3,'color','k')
hold on
plot(QTF,FQTF,':','Linewidth',3,'color','b')
plot(Qone,FQone,'Linewidth',3,'color','r')
set(gca,'FontSize',15)
set(gca,'linewidth',2)

leftaxis = 0;
rightaxis = 2;
upaxis = 4;
downaxis = 0;

%axis([leftaxis rightaxis downaxis upaxis])
legend('$\theta = 1/2$','$\theta = 3/4$','$\theta = 1$','Interpreter','latex')

legend('Location','northeast')
%hold on
%plot(Qhalf,QuadEnLShalf,'Linewidth',2,'color','b')
%plot(Qhalf,VoroEnLShalf,'Linewidth',2,'color','r')



