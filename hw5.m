function hw5()

% program may need to run a few times for points
% to plot correctly on each graph

close all;
format compact;

worldCoords = load('world_coords.txt');
imageCoords = load('image_coords.txt');

%worldCoords
%imageCoords

%[estCamMat, estReprErr] = estimateCameraMatrix(imageCoords, worldCoords);
%estCamMat
%estCamMat'
%whos('estReprErr')

%homoworldcords = [worldCoords ones(size(worldCoords(:,1)))];
%uv = [imageCoords ones(size(imageCoords(:,1)))];
%uv
%projMat = [1 0 0 0; 0 1 0 0; 0 0 1 0];

%xyz = [1 0 1; 5 3 0];
%uv = [771 427 1; 418 1115 1];
%uv(1,2)
%xyz(1,:).*uv(1,1)'
%P = [xyz(1,:) 1 0 0 0 0 -xyz(1,:).*uv(1,1) uv(1,1)*-1];
%P
P = zeros(30,12);
%A
c = 0;
for i = 1:15
    if rem(i,2)==1
        P(i+c,:) = [worldCoords(i,:) 1 0 0 0 0 -worldCoords(i,:).*imageCoords(i,1) imageCoords(i,1)*-1];
        P(i+1+c,:) = [0 0 0 0 worldCoords(i,:) 1 -worldCoords(i,:).*imageCoords(i,2) imageCoords(i,2)*-1];
    else
        P(i+1+c,:) = [worldCoords(i,:) 1 0 0 0 0 -worldCoords(i,:).*imageCoords(i,1) imageCoords(i,1)*-1];
        P(i+2+c,:) = [0 0 0 0 worldCoords(i,:) 1 -worldCoords(i,:).*imageCoords(i,2) imageCoords(i,2)*-1];
        c = c+2;
    end
end
%P
%whos('P')

U = P'*P;
%U
%whos('U')
[v1, d1] = eig(U);
%v1
%d1
minPair = eigs(U, 1, 'SM');
%minPair
%whos('minPair')
[V,D] = eigs(U, 1, 'SM');
V
%V
%D
%whos('V')
%whos('D')
bigM = [V(1:4)'; V(5:8)'; V(9:12)'];
bigM
%bigM

gridrep = imread('repliAx2.png');
%imageCoords(1,2)
figure, imshow(gridrep)
hold on
for i = 1:15
    %gridrep(imageCoords(i,2), imageCoords(i, 1), 1) = 255;
    %gridrep(imageCoords(i,2), imageCoords(i, 1), 2) = 0;
    %gridrep(imageCoords(i,2), imageCoords(i, 1), 3) = 0;
    plot(imageCoords(i, 1), imageCoords(i,2), 'r*')
    hold on
end
hold off

homoWorld = [worldCoords ones(size(worldCoords(:,1)))];
ThomoWorld = homoWorld';
uvw = bigM*ThomoWorld;
uv = uvw(1:2,:)./uvw(3,:);
tUV = uv';
hold off
figure, imshow(gridrep)
hold on
for i = 1:15
    %gridrep(imageCoords(i,2), imageCoords(i, 1), 1) = 255;
    %gridrep(imageCoords(i,2), imageCoords(i, 1), 2) = 0;
    %gridrep(imageCoords(i,2), imageCoords(i, 1), 3) = 0;
    plot(imageCoords(i, 1), imageCoords(i,2), 'r*')
    hold on
    plot(tUV(i, 1), tUV(i,2), 'b*')
    hold on
end
hold off


%THIS COMMENT STUB STATES THAT 
%THIS CODE IS THE PROPERTY OF OMAR R.G. (UofA Student)

%____________RMSERRORS_START________%
diff = imageCoords - tUV;
squared = diff.^2;
summed = sum(squared, 'all');
division = summed/15;
rms = sqrt(division);
rms
whos('rms')

%____________RMSERRORS_END________%

%_____compare with other M's__START

camMat1 = load('camera_matrix_1.txt');
camMat2 = load('camera_matrix_2.txt');

uvw1 = camMat1*ThomoWorld;
%uvw1
%uvw(2,1)/uvw(3,1)
uv1 = uvw1(1:2,:)./uvw1(3,:);
%uv1
tUV1 = uv1';
hold off
figure, imshow(gridrep)
hold on
for i = 1:15
    %gridrep(imageCoords(i,2), imageCoords(i, 1), 1) = 255;
    %gridrep(imageCoords(i,2), imageCoords(i, 1), 2) = 0;
    %gridrep(imageCoords(i,2), imageCoords(i, 1), 3) = 0;
    plot(imageCoords(i, 1), imageCoords(i,2), 'r*')
    hold on
    plot(tUV(i, 1), tUV(i,2), 'b*')
    hold on
    plot(tUV1(i, 1), tUV1(i,2), 'g*')
    hold on
end
hold off

%camMat2
uvw2 = camMat2*ThomoWorld;
%uvw2
%uvw(2,1)/uvw(3,1)
uv2 = uvw2(1:2,:)./uvw2(3,:);
%uv2
tUV2 = uv2';
hold off
figure, imshow(gridrep)
hold on
for i = 1:15
    %gridrep(imageCoords(i,2), imageCoords(i, 1), 1) = 255;
    %gridrep(imageCoords(i,2), imageCoords(i, 1), 2) = 0;
    %gridrep(imageCoords(i,2), imageCoords(i, 1), 3) = 0;
    plot(imageCoords(i, 1), imageCoords(i,2), 'r*')
    hold on
    plot(tUV(i, 1), tUV(i,2), 'b*')
    hold on
    plot(tUV1(i, 1), tUV1(i,2), 'g*')
    hold on
    plot(tUV2(i, 1), tUV2(i,2), 'y*')
    hold on
end
hold off

%_____compare with other M's__END___


%Graphics part B
% light coords: 33, 29, 44
%hold off
hold off
objectimg = imread('IMG_0861.jpeg');
hold off
%[x y z] = sphere;
%x
%y
%z
%R = 85; %change to 1/2 inch
figure, imshow(objectimg)
hold on
%surf(R*x+3,R*y+2,R*z+3);
%k = zeros(3,1);
%i=1;
r = 85; %roughly 1/2 inch in pixels derived from image
xo = 3;
yo = 2;
zo = 3;
theta=0:2*pi/240:2*pi ;
phi=-pi/2:pi/480:pi/2 ;
%{
[T,P] = meshgrid(theta,phi) ;
X = xo + r *cos(P).* cos(T);
Y = yo + r *cos(P).* sin(T);
Z = zo + r *sin(P) ;
hold on
%}

%{
plot3(X,Y,Z, 'r')
[nx,ny,nz] = surfnorm(X,Y,Z);
quiver3(X,Y,Z, nx, ny, nz)
hold off
%}


%pp = [9 14 11];
%xx = [X Y Z];
%whos('X')

%hold on
%camera location 9,14,11
lastMat = [];
tempMat = [];
for i = -pi/2:pi/480:pi/2
    for j = 0:pi/40:2*pi
        xnew = xo + r *cos(i).* cos(j);
        ynew = yo + r *cos(i).* sin(j);
        znew = zo + r *sin(i);
        tempMat = [xnew ynew znew];
        lastMat = [lastMat; tempMat];
    end
end
%lastMat
center = [3 2 3];
pp = [9 14 11];
hold on
for i = 1:38961
    nuex = [lastMat(i,1) lastMat(i,2) lastMat(i,3)];
    if(dot(pp-nuex, normalize(pp-center)) > 0)
        plot3(lastMat(i,1), lastMat(i,2), lastMat(i,3), 'b*')
        %dot(pp-nuex, normalize(nuex-center))
        hold on
    end
end
hold on
[lightx lighty lightz] = sphere();
surf(20*lightx-30, 20*lighty, 20*lightz)
plot3([-30; 3],[0; 2],[0; 3], 'r*-')
%lastMat
%lastMat(:,3)
%lastMat(:,1)
%lastMat(:,3)
hold off
hold off
%{
Zizo = zo + r *sin(-pi/2:pi/480:pi/2);
Yizo = yo + r .*cos(-pi/2:pi/480:pi/2).* sin(0:2*pi/480:2*pi);
Xizo = xo + r *cos(-pi/2:pi/480:pi/2).* cos(0:2*pi/480:2*pi);
whos('Zizo')
whos('Yizo')
whos('Xizo')
tLast = lastMat';
whos('tLast')
%}
%{
figure, 
hold on
plot3(Xizo, Yizo, Zizo)
hold off
%}

%{
for i = 1:241
    plot3(lastMat(i,1), lastMat(i,2), lastMat(i,3))
    hold on
end
%}
%{
hold off

nuex = lastMat(80,:);
whos('nuex')
whos('pp')
pp-nuex

normVec = normalize(nuex-center);
normVec
dotP = dot(pp-nuex,normalize(nuex-center));
dotP

whos('dotMat')

%figure,
hold on

hold off

whos('X')
whos('Z')

figure, plot3(lastMat(:,1), lastMat(:,2), lastMat(:,3));
hold on

%}


%light vector

%lastMat
%whos('lastMat')
%surf(R*x+3, R*y+2, R*z+3);
%hold off
%plot(x+3, y+2, z+3);
%plot(1,8);
%render sphere radius 1/2 inch 
%position (3,2,3) of any color
%{
xs = R.*sin(th).*cos(ph);
ys = R.*sin(th).*sin(ph);
zs = R.*cos(th);

x = x0 + cos(phi)*cos(theta)*R
y = y0 + cos(phi)*sin(theta)*R
z = z0 + sin(phi)*R
%}



%figure, sphere
%hold off




%UVW = homoWorld'.*bigM;


%uvw(1:4,:)
%uvw = bigM.*homoWorld;

%[R, t] = extrinsics(imageCoords, worldCoords(:,1:2), bigM);

%bigM = 
%P = worldCoords.*imageCoords;
%P
%calcM = imageCoords.*projMat.*worldCoords;
%calcM
%whos('calcM')

%m = [1 2 3 4 5 6 7 8 9 10 11 12]'
%m



end