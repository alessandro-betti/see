clear all; %#ok<CLALL>

workingDir = './';
destdir = './../_run_scripts/data/';
outputVideo = VideoWriter(fullfile(workingDir,'toybear.avi'),'Uncompressed AVI');
outputVideo.FrameRate = 25;
open(outputVideo);

% options
r = 30;
w = 200;
h = 250;
rotating_steps = 100;
frozen_steps = 100;

toy = imread([destdir 'toy.bmp']);
offsetx = w/2.0;
offsety = h/2.0;
hh = size(toy,1);
ww = size(toy,2);

angle = -pi/2.0;
jj = 0;
j = 1;
rep = 1;
toy_original = toy;
rot_on_itself = true;

for z = 1:rep*rotating_steps
    
    if ~rot_on_itself
        x = sqrt( (r*r) / (1.0 + tan(angle)^2) );
        y = - tan(angle) * x;

        if j > round(rotating_steps/2.0)
            x = -x;
            y = -y;
        end
    else   
        toy = imrotate(toy_original,rad2deg(pi/2.0+angle));
        mask = imrotate(ones(size(toy_original)),rad2deg(pi/2.0+angle));
        mask = mask == 1;
        toy(~mask) = 255.0;
        [hh,ww,~] = size(toy);
        y = offsety;
        x = offsetx;
    end
    
    a = round(y - hh/2.0);
    b = a + hh - 1;
    c = round(x - ww/2.0);
    d = c + ww - 1;
    
    img = uint8(ones(h,w,3) * 255.0);
    img(a:b,c:d,1:3) = toy;
    
    writeVideo(outputVideo, img);
  
    angle = angle + (2.0*pi)/rotating_steps;
  
    if j == rotating_steps
        for u = 1:frozen_steps
            writeVideo(outputVideo, img);
        end        
        if jj == rep
            break
        else
            j = 1;
            jj = jj + 1;
            angle = -pi/2.0;
        end
    else
        j = j + 1;
    end
end

close(outputVideo);
