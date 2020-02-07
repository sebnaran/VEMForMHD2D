function[Bx,By,Mx,My,E,Ex,Ey] = GetData(file)
%file = 'out.txt';
    text = fileread(file);

    C = split(text,'|');


    Bx = C(1);
    Bx = erase(Bx,',');
    Bx = str2num(Bx{1});


    By = C(2);
    By = erase(By,',');
    By = str2num(By{1});


    Mx = C(3);
    Mx = erase(Mx,',');
    Mx = str2num(Mx{1});


    My = C(4);
    My = erase(My,',');
    My = str2num(My{1});


    E = C(5);
    E = erase(E,',');
    E = str2num(E{1});

    Ex = C(6);
    Ex = erase(Ex,',');
    Ex = str2num(Ex{1});

    Ey = C(7);
    Ey = erase(Ey,',');
    Ey = str2num(Ey{1});
 

