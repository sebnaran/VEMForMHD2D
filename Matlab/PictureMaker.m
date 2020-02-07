

    i= 0.35;
    %This opens the files
    k = sprintfc('%03.3f',i);
    k = k{1};
    file = join({num2str(k),'.txt'});
    file = file{1};
    file = erase(file,' ');
    [Bx,By,Mx,My,E,Ex,Ey] = GetData(file);
    
    %figure(1)
    
    %Newx = vec2mat(Ex,24);
    %Newy = vec2mat(Ey,24);
    %E = vec2mat(E,24);
    
    %surf(Newx,Newy,E)
   
    
    %colorbar
    %axis([-1.2 1.2 -1.2 1.2])
    %title(join({'Electric Field','T=',num2str(i)}))
    %view(3)
    
    
    figure(1);
    clf
 
    
    %quiver(Mx(1:6:529),My(1:6:529),Bx(1:6:529),By(1:6:529))
    quiver(Mx(1:3:529),My(1:3:529),Bx(1:3:529),By(1:3:529))
    axis([-1.2 1.2 -1.2 1.2])
    title(join({'MagneticField','T=',num2str(i)}))
   
    
   
    
    Mx = vec2mat(Mx,23);
    My = vec2mat(My,23);
    Bx = vec2mat(Bx,23);
    By = vec2mat(By,23);
    
    
    %starty = -1:0.1:1;
    %startx = -0.8*ones(size(starty));
    starty = My(1,:)';
    startx = 0.01+Mx(1,:)';
    starty(length(starty)) = [];
    startx(length(startx)) = [];
    for k=1:12
        startx(k) = -startx(k);
    end
    
    figh = figure(1);
    
    streamline(Mx',My',Bx',By',startx,starty)
    
    axis([-1.2 1.2 -1.2 1.2])
    title(join({'MagneticField','T=',num2str(i)}))
    
    pos = get(figh,'position');
    
    set(figh,'position',[pos(1:2)/4 pos(3:4)*2])
    
    
    
    
    
    
    
    
    
    
    
    
%    if i == 0
%        figure(4)
%         [vx,vy] = velocity(Mx,My);
%         startx = -.9:0.1:0.9;
%         starty = -0.9*ones(length(startx),1);
%         streamline(Mx',My',vx',vy',startx,starty)
%         starty = -starty;
%         streamline(Mx',My',vx',vy',startx,starty)
%         streamline(Mx',My',vx',vy',[-0.01],[0])
%         streamline(Mx',My',vx',vy',[0.01],[0])
%         axis([-1.2 1.2 -1.2 1.2])
%        title('Velocity Field')
%         figure(5)
%        quiver(Mx,My,vx,vy)
%        axis([-1.2 1.2 -1.2 1.2])
%        title('Velocity Field')
%    end
    
    
    
