%--------------------Part I--------------------
image = [];
%read 12 model images and "SwainCollageForBackprojectionTesting.bmp"
image = cat(4,image,imread('swain_database/1/bakit.sqr.128.bmp'));
image = cat(4,image,imread('swain_database/1/girls.sqr.128.bmp'));
image = cat(4,image,imread('swain_database/1/balloons.sqr.128.bmp'));
image = cat(4,image,imread('swain_database/1/car.sqr.128.bmp'));
image = cat(4,image,imread('swain_database/1/carebears.sqr.128.bmp'));
image = cat(4,image,imread('swain_database/1/charmin.sqr.128.bmp'));
image = cat(4,image,imread('swain_database/1/clamchowder.sqr.128.bmp'));
image = cat(4,image,imread('swain_database/1/chickensoupnoodles.sqr.128.bmp'));
image = cat(4,image,imread('swain_database/1/crunchberries.sqr.128.bmp'));
image = cat(4,image,imread('swain_database/1/flyer.sqr.128.bmp'));
image = cat(4,image,imread('swain_database/1/frankenberry.sqr.128.bmp'));
image = cat(4,image,imread('swain_database/1/garan.sqr.128.bmp'));
collage = imread('SwainCollageForBackprojectionTesting.bmp');


for i = 1:12
    model = image(:,:,:,i);

    %turn images to color histogram
    rgb_hist1 = img_to_3dhist(model);
    rgb_hist2 = img_to_3dhist(collage);
    %calculate ratio histogram
    R = min(rgb_hist1./rgb_hist2, 1);
    %back project onto collage image
    back_pro = back_projection(R,collage);
    mask = create_mask(35); 
    %convolve the back projected image
    con=conv2(mask,back_pro); 
    %locate the maximum intensity after convolution
    [val, idx] = max(con(:));
    %find area with 90% of max intensity
    y=find(con>val*0.9);
    res=zeros(size(con));  
	res(y)=1;   
    %filter object with the largest area
    res = bwpropfilt(im2bw(res),'Area',1);
    %find the center point of the object
    rprops = regionprops(res,'Centroid'); 
    %mark on original collage image
    figure('Position',[100, 100, 1000, 400]);
    subplot(3,6,7);
    imshow(model);
    title('Model image','FontSize',20);
    subplot(3,6,[3,4,5,6,9,10,11,12,15,16,17,18]);
    imshow(collage);
    title('Location on collage image mark with yellow star','FontSize',20);
    hold on;
    plot(rprops.Centroid(1)-35, rprops.Centroid(2)-35,'gp','LineWidth',4,'MarkerSize',45);
    hold off;
input('Hit any key to continue.');
close;
end

clear all;

%--------------------Part II--------------------
video = load('CMPT412_bluecup.mat');
video = video.bluecup;
marked_video = [];
figure;imshow(video(:,:,:,1));
%mouse input select two diagonal points
[x,y] = ginput(2);
close;
%calculate width and height
region_width = x(2)-x(1);
region_height = y(2)-y(1);
%crop and save the image
region_image = imcrop(video(:,:,:,1),[min(x),min(y),region_width,region_height]);
%figure;imshow(region_image);
rgb_hist_region = img_to_3dhist(region_image);
r = min([region_width,region_height])/2;
mask = create_mask(r);
%interate each frame
for f = 1:size(video,4)
    figure;
    imshow(video(:,:,:,f));
    frame_hist_region = img_to_3dhist(video(:,:,:,f));
    R = min(rgb_hist_region./frame_hist_region, 1);
    %back project onto collage image
    back_pro = back_projection(R,video(:,:,:,f));
    %convolve the back projected image
    con=conv2(mask,back_pro);
    %locate the maximum intensity after convolution
    [val, idx] = max(con(:));
    %find area with 90% of max intensity
    y=find(con>val*0.9);
    res=zeros(size(con));  
    res(y)=1;   
    %filter object with the largest area
    res = bwpropfilt(im2bw(res),'Area',1);
    %find the center point of the object
    rprops = regionprops(res,'Centroid'); 
    %draw centroid in each frame
    hold;
    plot(rprops.Centroid(1)-r, rprops.Centroid(2)-r,'yp','LineWidth',4,'MarkerSize',45);
    F = getframe;
    marked_video = cat(4,marked_video,F.cdata);
    if (mod(f,5) == 0)
        imwrite(F.cdata, join(['frame_',int2str(f),'.jpg']));
    end
    close;
end
v_writer = VideoWriter('marked_bluecup.avi');
v_writer.FrameRate = 15;
open(v_writer);
writeVideo(v_writer, marked_video);
close(v_writer);
clear all;

input('Hit any key to continue.');

%--------------------Part III--------------------
video = load('CMPT412_blackcup.mat');
video = video.blackcup;
marked_video = [];
figure;imshow(video(:,:,:,1));
[frame_height, frame_width, p] = size(video(:,:,:,1));
%mouse input select two diagonal points
[x,y] = ginput(2);
close;
%calculate width and height
region_width = x(2)-x(1);
region_height = y(2)-y(1);
%crop and save the image
region_image = imcrop(video(:,:,:,1),[min(x),min(y),region_width,region_height]);
%figure;imshow(region_image);
rgb_hist_region = img_to_3dhist(region_image);
r = max([region_width,region_height]);
mask = create_mask(r);
%find the center pixel in the region of interest
center_x = round(x(2)-region_width/2);
center_y = round(y(2)-region_height/2);
%interate each frame
for f = 1:size(video,4)
    figure;
    imshow(video(:,:,:,f));
    frame_hist_region = img_to_3dhist(video(:,:,:,f));
    R = min(rgb_hist_region./frame_hist_region, 1);
    %back project onto collage image
    back_pro = back_projection(R,video(:,:,:,f));
    check = true;
    while check
        new_x = 0;
        new_y = 0;
        kernel = 0;
        %look at region within the square of 2 radius
        for x_i = round(center_x-r):round(center_x+r)
            for y_i = round(center_y-r):round(center_y+r)
                %check if the distance between the pixel and center is smaller
                %than radius or if the index is out of range
                if sqrt((x_i-center_x)^2+(y_i-center_y)^2) < r && x_i>0 && y_i>0 && x_i<frame_width && y_i<frame_height
                    %add up the weighted value
                    new_x = new_x + (back_pro(y_i,x_i)*x_i);
                    new_y = new_y + (back_pro(y_i,x_i)*y_i);
                    %add up the weight
                    kernel = kernel + back_pro(y_i,x_i);
                end
            end
        end
        %calculate the new mean
        new_x = new_x/kernel;
        new_y = new_y/kernel;
        check = (abs(new_x-center_x) > 5) || (abs(new_y-center_y) > 5);
        center_x = new_x;
        center_y = new_y;
    end
    %draw centroid in each frame
    hold;
    plot(center_x, center_y,'yp','LineWidth',4,'MarkerSize',45);
    F = getframe;
    marked_video = cat(4,marked_video,F.cdata);
    if (mod(f,5) == 0)
        imwrite(F.cdata, join(['frame_black_',int2str(f),'.jpg']));
    end
    close;
end
v_writer = VideoWriter('marked_blackcup.avi');
v_writer.FrameRate = 15;
open(v_writer);
writeVideo(v_writer, marked_video);
close(v_writer);
%--------------------Functions--------------------
%back projection function
function back_pro = back_projection(R, image)
    [height,width,P] = size(image);
    back_pro = zeros(height, width);
    for x = 1:width
        for y = 1:height
            %replace each pixel with corresponding value in ratio histogram
            bin = floor(double(image(y,x,:))/32)+1;
            back_pro(y,x) = R(bin(1),bin(2),bin(3));
        end
    end
end
%create color histogram
function rgb_hist = img_to_3dhist(image)
    rgb_hist = zeros(8,8,8); 
    [height,width,P] = size(image);
    for x = 1:width
        for y = 1:height
            %count color and add into corresponding bin
            rgb = floor(double(image(y,x,:))/32)+1;
            rgb_hist(rgb(1),rgb(2),rgb(3)) = rgb_hist(rgb(1),rgb(2),rgb(3))+1;
        end
    end
end
%create a circular mask, using partial code from sample circlefinder.m
function mask = create_mask(r)
    w = r*2;
    [x, y] = meshgrid(1:w, 1:w);
    circle = ((x - (w/2)).^2 + (y - (w/2)).^2 <= r^2);
    mask=double(circle); 
    mask(mask==0)=-1;  
end
