function processFixedPipeline(inputFolder, outputFolder)
% Apply all 8 techniques to every image (fixed pipeline)

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
        
        % Apply all techniques in sequence
        processed = img;
        processed = rgb2gray(processed);           % 1. Grayscale
        processed = histeq(processed);             % 2. Histogram equalization
        processed = imgaussfilt(processed, 2);     % 3. Gaussian smoothing
        
        % Convert to binary for remaining operations
        T = adaptthresh(processed, 0.4);
        processed = imbinarize(processed, T);      % 4. Adaptive thresholding
        processed = edge(processed, 'Canny');      % 5. Canny edge detection
        
        % Morphological operations
        se = strel('disk', 3);
        processed = imopen(processed, se);         % 6. Opening
        processed = imclose(processed, se);        % 7. Closing
        
        % Convert back to uint8 for saving
        processed = uint8(processed) * 255;
        
        % Save processed image
        outputName = sprintf('%s_processed.png', baseName);
        imwrite(processed, fullfile(outputFolder, outputName));
    end
    
    fprintf('Fixed pipeline complete: %d images processed\n', length(imgFiles));
end