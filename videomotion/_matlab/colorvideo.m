clear variables;

dirs = {
    './exp/skater/frames/00000001/', ...
    './exp/skater/frames/00000002/', ...
    './exp/skater/frames/00000003/'
    };

destdir = './exp/skater/new/';

k = 1;
for j = 1:length(dirs)
    for i = 1:100
        if ~exist([dirs{j} sprintf('%03d', i) '.png'], 'file')
            break
        end
        I = repmat(rgb2gray(imread([dirs{j} sprintf('%03d', i) '.png'])),1,1,3);
        I(:,:,2) = 0;
        I(:,:,3) = 0;
        imwrite(I, [destdir sprintf('%03d', k) '.png']);
        k = k + 1;
    end
end
for j = 1:length(dirs)
    for i = 1:100
        if ~exist([dirs{j} sprintf('%03d', i) '.png'], 'file')
            break
        end
        I = repmat(rgb2gray(imread([dirs{j} sprintf('%03d', i) '.png'])),1,1,3);
        I(:,:,1) = 0;
        I(:,:,3) = 0;
        imwrite(I, [destdir sprintf('%03d', k) '.png']);
        k = k + 1;
    end
end
for j = 1:length(dirs)
    for i = 1:100
        if ~exist([dirs{j} sprintf('%03d', i) '.png'], 'file')
            break
        end
        I = repmat(rgb2gray(imread([dirs{j} sprintf('%03d', i) '.png'])),1,1,3);
        I(:,:,1) = 0;
        I(:,:,2) = 0;
        imwrite(I, [destdir sprintf('%03d', k) '.png']);
        k = k + 1;
    end
end

