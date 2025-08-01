function processSingleTechnique(inputFolder, outputFolder, technique)
% Apply only one technique for individual testing

    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    
    imgFiles = [dir(fullfile(inputFolder, '*.jpg')); dir(fullfile(inputFolder, '*.png'))];
    
    for k = 1:length(imgFiles)
        imgPath = fullfile(inputFolder, imgFiles(k).name);
        img = imread(imgPath);
        
        % Strip alpha if present
        if size(img, 3) == 4
            img = img(:, :, 1:3);
        end
        
        [~, baseName, ~] = fileparts(imgFiles(k).name);
        
        % Apply single technique
        processed = applySingleTech(img, technique);
        
        % Save processed image
        outputName = sprintf('%s_processed.png', baseName);
        imwrite(processed, fullfile(outputFolder, outputName));
    end
    
    fprintf('Single technique "%s" complete: %d images processed\n', technique, length(imgFiles));
end

function processed = applySingleTech(img, technique)
% Apply one specific technique

    switch lower(technique)
        case 'histogram_eq'
            gray = rgb2gray(img);
            processed = histeq(gray);
            
        case 'gaussian_smooth'
            gray = rgb2gray(img);
            processed = imgaussfilt(gray, 2);
            
        case 'canny_edge'
            gray = rgb2gray(img);
            processed = edge(gray, 'Canny');
            processed = uint8(processed) * 255;
            
        case 'adaptive_thresh'
            gray = rgb2gray(img);
            T = adaptthresh(gray, 0.4);
            processed = imbinarize(gray, T);
            processed = uint8(processed) * 255;
            
        otherwise
            % Default to grayscale
            processed = rgb2gray(img);
    end
    
    % Ensure uint8 output
    if ~isa(processed, 'uint8')
        processed = im2uint8(processed);
    end
end