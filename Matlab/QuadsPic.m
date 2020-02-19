x     = -1:0.2:1;
y     = -1:0.2:1;
[X,Y] = meshgrid(x,y);
F     = 0*X;
for i = 1:length(x)
    for j = 1:length(y)
        %C(i,j) = [1,1,1];
        if i>1 & j>1 & i<11 & j<11	
            X(i,j) = X(i,j)+0.06*(2*rand-1);
            Y(i,j) = Y(i,j)+0.06*(2*rand-1);
        end
    end
end
C(:,:,1) = ones(11);
C(:,:,2) = ones(11);
C(:,:,3) = ones(11);
figure(3)
Quads = surf(X,Y,F,C);
set(Quads,'linewidth',4)
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);
view(0,90)
leftaxis  = -1;
rightaxis = 1;
upaxis    = 1;
downaxis  = -1;

axis([leftaxis rightaxis downaxis upaxis])
pbaspect([1 1 1])
        
        

%export with width = 25cms and height=15cms