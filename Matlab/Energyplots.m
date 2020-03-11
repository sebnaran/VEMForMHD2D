clear all
fL
fR
fF
%N = [0:10];
N = [0:10];

figure(1)
clf
plot(N,F,'Linewidth',2,'color','k')
hold on
plot(N,R,'--','Linewidth',2,'color','b')
plot(N,L,':','Linewidth',2,'color','r')
set(gca,'FontSize',15)
set(gca,'linewidth',2)
legend('$\mathcal{E}$','$\mathcal{E}_R$', '$\mathcal{E}_L$','interpreter','latex')
legend('Location','northwest')
