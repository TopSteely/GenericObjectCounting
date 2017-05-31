% Compile anisotropic gaussian filter
if(~exist('SelectiveSearchCodeIJCV/anigauss'))
    mex SelectiveSearchCodeIJCV/Dependencies/anigaussm/anigauss_mex.c SelectiveSearchCodeIJCV/Dependencies/anigaussm/anigauss.c -output SelectiveSearchCodeIJCV/anigauss
end

if(~exist('SelectiveSearchCodeIJCV/mexCountWordsIndex'))
    mex SelectiveSearchCodeIJCV/Dependencies/mexCountWordsIndex.cpp
end

% Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
if(~exist('SelectiveSearchCodeIJCV/mexFelzenSegmentIndex'))
    mex SelectiveSearchCodeIJCV/Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp -output SelectiveSearchCodeIJCV/mexFelzenSegmentIndex;
end

%%
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorType = colorTypes{1}; % Single color space for demo

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
k = 200; % controls size of segments of initial segmentation. 
minSize = k;
sigma = 0.8;
images = '/home/t/Downloads/ucsdpeds1/video/test/vidf1_33_%s.y/vidf1_33_%s_f%s.png';
boxes_ = '/home/t/Downloads/ucsdpeds1/SS_Boxes/%s_%s.txt';

tic
index = 1;
for i = 0:1
    for j = 1:200
        if exist(sprintf(images, num2str(i,'%03d' ),num2str(i,'%03d' ),num2str(j,'%03d' )), 'file')
            [im,map] = imread(sprintf(images, num2str(i,'%03d' ),num2str(i,'%03d' ),num2str(j,'%03d' )));
            rgbImage = cat(3, im, im, im);
        else
            sprintf(images, num2str(i,'%03d' ),num2str(i,'%03d' ),num2str(j,'%03d' ))
            continue
        end
        
        % Perform Selective Search
        [boxes, ~ ,~ ,~] = Image2HierarchicalGrouping(rgbImage, sigma, k, minSize, colorType, simFunctionHandles);
        boxes = BoxRemoveDuplicates(boxes);
        % get ground truth bounding boxes from annotations
        for b = 1:size(boxes, 1)
    %         figure;
    %         imshow(im)
            proposal = [boxes(b, 2) boxes(b, 1) boxes(b, 4)-boxes(b, 2) boxes(b, 3)-boxes(b, 1)];
            % -1 because of matlab->python
            if b == 1
                pb = [proposal(1) proposal(2) proposal(3)+proposal(1) proposal(4)+proposal(2)]-1;
            else
                pb = [pb;[proposal(1) proposal(2) proposal(3)+proposal(1) proposal(4)+proposal(2)]-1];
            end
        
        end 
        % output bounding boxes
        csvwrite(sprintf(boxes_, num2str(i,'%03d' ),num2str(j,'%03d' )), pb)
    end
end
toc
