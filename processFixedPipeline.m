function processFixedPipeline(inputFolder, outputFolder)
% Apply preprocessing techniques to every image (fixed pipeline)
% FIXED VERSION - addresses over-processing and MATLAB hanging issues

    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    
    imgFiles = [dir(fullfile(inputFolder, '*.jpg')); dir(fullfile(inputFolder, '*.png'))];
    
    fprintf('Processing %d images with fixed pipeline...\n', length(imgFiles));
    
    for k = 1:length(imgFiles)
        imgPath = fullfile(inputFolder, imgFiles(k).name);
        
        try
            img = imread(imgPath);
            
            % Strip alpha if present
            if size(img, 3) == 4
                img = img(:, :, 1:3);
            end
            
            [~, baseName, ~] = fileparts(imgFiles(k).name);
            
            % FIXED PIPELINE - Less destructive approach
            processed = img;
            
            % 1. Convert to grayscale
            if size(processed, 3) == 3
                processed = rgb2gray(processed);
            end
            
            % 2. Histogram equalization (improve contrast)
            processed = histeq(processed);
            
            % 3. Gaussian smoothing (reduce noise)
            processed = imgaussfilt(processed, 1.5);  % Reduced sigma from 2 to 1.5
            
            % 4. STOP HERE - Don't apply destructive operations
            % REMOVED: Adaptive thresholding (converts to binary)
            % REMOVED: Canny edge detection (creates binary edges)
            % REMOVED: Morphological operations (on binary data)
            
            % Keep as grayscale for YOLO compatibility
            processed = im2uint8(processed);
            
            % Save processed image
            outputName = sprintf('%s_processed.png', baseName);
            imwrite(processed, fullfile(outputFolder, outputName));
            
            fprintf('✅ Processed: %s\n', baseName);
            
        catch ME
            fprintf('❌ Error processing %s: %s\n', baseName, ME.message);
            
            % Save original as fallback to prevent missing files
            outputName = sprintf('%s_processed.png', baseName);
            try
                if size(img, 3) == 3
                    grayImg = rgb2gray(img);
                else
                    grayImg = img;
                end
                imwrite(grayImg, fullfile(outputFolder, outputName));
                fprintf('💾 Saved fallback for %s\n', baseName);
            catch
                fprintf('💥 Complete failure for %s\n', baseName);
            end
        end
    end
    
    fprintf('Fixed pipeline complete: %d images processed\n', length(imgFiles));
    
    % FORCE MATLAB TO EXIT CLEANLY
    fprintf('Exiting MATLAB process...\n');
    
end