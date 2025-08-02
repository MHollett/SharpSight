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
        
        try
            % Apply single technique
            processed = applySingleTech(img, technique);
            
            % Save processed image
            outputName = sprintf('%s_processed.png', baseName);
            imwrite(processed, fullfile(outputFolder, outputName));
            
            fprintf('Processed %s with %s\n', baseName, technique);
            
        catch ME
            fprintf('Error processing %s with %s: %s\n', baseName, technique, ME.message);
            
            % Save grayscale as fallback
            outputName = sprintf('%s_processed.png', baseName);
            if size(img, 3) == 3
                grayImg = rgb2gray(img);
            else
                grayImg = img;
            end
            imwrite(grayImg, fullfile(outputFolder, outputName));
        end
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
            edges = edge(gray, 'Canny');
            processed = uint8(edges) * 255;
            
        case 'adaptive_thresh'
            gray = rgb2gray(img);
            T = adaptthresh(gray, 0.4);
            binary = imbinarize(gray, T);
            processed = uint8(binary) * 255;
            
        otherwise
            % Default to grayscale
            processed = rgb2gray(img);
    end
    
    % Ensure uint8 output
    if ~isa(processed, 'uint8')
        if islogical(processed)
            processed = uint8(processed) * 255;
        else
            processed = im2uint8(processed);
        end
    end
    
    % Ensure the image is valid
    if isempty(processed) || any(size(processed) == 0)
        % Fallback to grayscale
        processed = rgb2gray(img);
        if ~isa(processed, 'uint8')
            processed = im2uint8(processed);
        end
    end
end