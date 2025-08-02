function processDroneDataset(inputFolder, outputFolder)

    % --- Configuration ---
    cfg.noiseThreshold = 0.01;
    cfg.contrastThreshold = 0.15;
    cfg.lowContrastHSV = 0.1;
    cfg.lightingVarThreshold = 0.03;
    cfg.edgeThickness = 1;                
    cfg.cannyThresholds = [0.12 0.3];     
    cfg.noiseFractionThreshold = 0.03;
    cfg.gapFractionThreshold = 0.015;
    cfg.houghRadiusRange = [8 50];
    cfg.pyramidScales = [1, 0.5, 0.25];

    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end

    % --- Support JPG + PNG ---
    imgFiles = [dir(fullfile(inputFolder, '*.jpg')); dir(fullfile(inputFolder, '*.png'))];

    for k = 1:length(imgFiles)
        imgPath = fullfile(inputFolder, imgFiles(k).name);
        img = imread(imgPath);

        % --- Strip alpha if PNG with transparency ---
        if size(img,3) == 4
            img = img(:,:,1:3);
        end

        [pyramidImages, visImg, stages, metrics, pyramidBoxes, scales] = dynamicPipelineAI(img, cfg);

        [~, baseName, ~] = fileparts(imgFiles(k).name);

        % Save pyramid levels and annotations
        for s = 1:length(scales)
            scaleSuffix = strrep(num2str(scales(s)), '.', '_');
            imgName = sprintf('enhanced_%s_scale%s.png', baseName, scaleSuffix);
            imwrite(pyramidImages{s}, fullfile(outputFolder, imgName));

            if ~isempty(pyramidBoxes{s})
                labelName = sprintf('enhanced_%s_scale%s.txt', baseName, scaleSuffix);
                fid = fopen(fullfile(outputFolder, labelName), 'w');
                for i = 1:size(pyramidBoxes{s},1)
                    fprintf(fid, '0 %.6f %.6f %.6f %.6f\n', pyramidBoxes{s}(i,:));
                end
                fclose(fid);
            end
        end

        % Save visualization/debug image
        imwrite(visImg, fullfile(outputFolder, ['vis_' baseName '.png']));

        % Save metadata
        metaFile = fullfile(outputFolder, [baseName '_meta.txt']);
        fid = fopen(metaFile, 'w');
        fprintf(fid, "Pipeline Stages:\n");
        for s = 1:length(stages)
            fprintf(fid, "- %s\n", stages{s});
        end

        fprintf(fid, "\nImage Metrics:\n");
        fprintf(fid, "Contrast: %.6f\n", metrics.contrast);
        fprintf(fid, "Noise: %.6f\n", metrics.noise);
        fprintf(fid, "Edge Density: %.6f\n", metrics.edgeDensity);
        fprintf(fid, "Lighting Variance: %.6f\n", metrics.lightingVar);

        fprintf(fid, "\nFiltering:\n");
        fprintf(fid, "Gaussian Sigma Used: %.3f\n", metrics.gaussianSigma);

        fprintf(fid, "\nMorphology Metrics:\n");
        fprintf(fid, "Noise Fraction: %.6f\n", metrics.noiseFraction);
        fprintf(fid, "Gap Fraction: %.6f\n", metrics.gapFraction);
        fprintf(fid, "Morphology Operation: %s\n", metrics.morphologyOperation);

        fprintf(fid, "\nEdge Detection:\n");
        fprintf(fid, "Canny White Edge Outlines Applied: Yes\n");
        fprintf(fid, "Edge Thickness: %d px\n", cfg.edgeThickness);
        fprintf(fid, "Canny Thresholds: [%.2f %.2f]\n", cfg.cannyThresholds(1), cfg.cannyThresholds(2));

        fprintf(fid, "\nCircular Hough Transform:\n");
        fprintf(fid, "Total Circles Detected (all scales): %d\n", metrics.totalCircles);

        fprintf(fid, "\nPyramid:\n");
        fprintf(fid, "Scales Used: %s\n", mat2str(scales));

        fclose(fid);
    end

    disp('Dataset processing complete (Canny with configurable thresholds and white outlines).');

end


function [pyramidImages, visImg, pipelineStages, metrics, pyramidBoxes, scales] = dynamicPipelineAI(inputImg, cfg)

    pipelineStages = {};
    img = im2double(inputImg);

    % --- Analysis ---
    gray = rgb2gray(img);
    contrastMetric = std2(gray);
    noiseMetric = estimateNoise(gray);
    edgeMetric = mean(edge(gray, 'Canny'), 'all');
    lightingVar = stdfilt(gray);

    metrics.contrast = contrastMetric;
    metrics.noise = noiseMetric;
    metrics.edgeDensity = edgeMetric;
    metrics.lightingVar = mean(lightingVar, 'all');
    metrics.gaussianSigma = 0;

    visSteps = {};
    stageLabels = {};
    visSteps{end+1} = img;
    stageLabels{end+1} = 'Original Image';

    % --- Color space conversion ---
    if contrastMetric < cfg.lowContrastHSV || metrics.lightingVar > cfg.lightingVarThreshold*1.5
        hsvImg = rgb2hsv(img);
        imgToProcess = hsvImg(:,:,3);
        pipelineStages{end+1} = 'HSV (V-channel)';
    else
        imgToProcess = gray;
        pipelineStages{end+1} = 'Grayscale';
    end
    visSteps{end+1} = imgToProcess;
    stageLabels{end+1} = pipelineStages{end};

    % --- Histogram equalization ---
    if contrastMetric < cfg.contrastThreshold
        imgToProcess = histeq(imgToProcess);
        pipelineStages{end+1} = 'Histogram Equalization';
        visSteps{end+1} = imgToProcess;
        stageLabels{end+1} = pipelineStages{end};
    end

    % --- Adaptive Gaussian smoothing ---
    if noiseMetric > cfg.noiseThreshold
        sigma = min(1.0, max(0.3, noiseMetric * 8));
        imgToProcess = imgaussfilt(imgToProcess, sigma, 'FilterSize', 3);

        metrics.gaussianSigma = sigma;
        pipelineStages{end+1} = sprintf('Gaussian Smoothing (σ=%.2f)', sigma);
        visSteps{end+1} = imgToProcess;
        stageLabels{end+1} = pipelineStages{end};
    end

    % --- Grayscale Morphology ---
    imgForMetrics = im2double(imgToProcess);
    if islogical(imgForMetrics)
        imgForMetrics = mat2gray(imgForMetrics);
    end
    tempMask = imbinarize(imgForMetrics, graythresh(imgForMetrics));

    noiseFraction = sum(bwareaopen(tempMask, 2, 8),'all') / numel(tempMask);
    filledMask = imfill(tempMask, 'holes');
    gapFraction = sum(filledMask(:) - tempMask(:)) / numel(tempMask);

    se = strel('disk', 3);

    if noiseFraction >= cfg.noiseFractionThreshold && noiseFraction > gapFraction
        imgToProcess = imopen(imgToProcess, se);
        morphOp = 'Grayscale Opening';
        pipelineStages{end+1} = morphOp;
        visSteps{end+1} = imgToProcess;
        stageLabels{end+1} = morphOp;

    elseif gapFraction >= cfg.gapFractionThreshold && gapFraction > noiseFraction
        imgToProcess = imclose(imgToProcess, se);
        morphOp = 'Grayscale Closing';
        pipelineStages{end+1} = morphOp;
        visSteps{end+1} = imgToProcess;
        stageLabels{end+1} = morphOp;

    elseif noiseFraction >= cfg.noiseFractionThreshold && gapFraction >= cfg.gapFractionThreshold
        imgToProcess = imclose(imopen(imgToProcess, se), se);
        morphOp = 'Grayscale Opening + Closing';
        pipelineStages{end+1} = morphOp;
        visSteps{end+1} = imgToProcess;
        stageLabels{end+1} = morphOp;
    else
        morphOp = 'None';
    end

    metrics.noiseFraction = noiseFraction;
    metrics.gapFraction = gapFraction;
    metrics.morphologyOperation = morphOp;

    % --- Always Canny Edge Detection with White Outlines ---
    edges = edge(imgToProcess, 'Canny', cfg.cannyThresholds);

    % Optional edge thickness (dilate edges if > 1)
    if cfg.edgeThickness > 1
        seThick = strel('disk', cfg.edgeThickness-1);
        edges = imdilate(edges, seThick);
    end

    % Ensure RGB for overlay
    if size(imgToProcess,3) == 1
        baseImg = repmat(imgToProcess, [1 1 3]);
    else
        baseImg = imgToProcess;
    end

    edgeOverlay = baseImg;

    % Draw white edges
    for ch = 1:3
        channel = edgeOverlay(:,:,ch);
        channel(edges) = 1;
        edgeOverlay(:,:,ch) = channel;
    end

    imgToProcess = edgeOverlay;
    pipelineStages{end+1} = sprintf('Canny White Edge Outlines (Thresh=%.2f/%.2f, Thick=%d)', ...
        cfg.cannyThresholds(1), cfg.cannyThresholds(2), cfg.edgeThickness);
    visSteps{end+1} = imgToProcess;
    stageLabels{end+1} = pipelineStages{end};

    % --- Multi-scale pyramid + Hough ---
    scales = cfg.pyramidScales;
    pyramidImages = cell(length(scales),1);
    pyramidBoxes = cell(length(scales),1);
    totalCircles = 0;

    for s = 1:length(scales)
        scaledImg = imresize(imgToProcess, scales(s));
        pyramidImages{s} = im2uint8(scaledImg);

        if islogical(edges)
            [centers, radii] = imfindcircles(edges, cfg.houghRadiusRange);
            totalCircles = totalCircles + size(centers,1);

            if ~isempty(centers)
                [h, w, ~] = size(scaledImg);
                boxes = [];
                for i = 1:size(centers,1)
                    cx = centers(i,1) / w;
                    cy = centers(i,2) / h;
                    bw = (2*radii(i)) / w;
                    bh = (2*radii(i)) / h;
                    boxes = [boxes; cx cy bw bh];
                end
                pyramidBoxes{s} = boxes;
            else
                pyramidBoxes{s} = [];
            end
        else
            pyramidBoxes{s} = [];
        end
    end

    metrics.totalCircles = totalCircles;
    metrics.pyramidScales = scales;
    metrics.generatedPyramidLevels = length(scales);

    for s = 1:length(scales)
        stageLabels{end+1} = sprintf('Pyramid %.2fx', scales(s));
    end

    % --- Visualization ---
    visSteps = [visSteps(:)', pyramidImages(:)'];
    visImg = buildVisualizationGrid(visSteps, stageLabels);

end


function n = estimateNoise(img)
    H = fspecial('laplacian', 0);
    n = std2(imfilter(img,H,'replicate'));
end


function visImg = buildVisualizationGrid(steps, labels)
    numSteps = length(steps);
    cols = ceil(sqrt(numSteps));
    rows = ceil(numSteps/cols);

    % Build montage first without labels
    for i = 1:numSteps
        steps{i} = im2uint8(steps{i});
    end
    visImgStruct = montage(steps, 'Size', [rows cols]);
    visImg = visImgStruct.CData;

    if nargin > 1 && ~isempty(labels)
        tileHeight = size(visImg,1) / rows;
        tileWidth  = size(visImg,2) / cols;

        for idx = 1:numSteps
            r = floor((idx-1)/cols);
            c = mod((idx-1), cols);
            xPos = c * tileWidth + 10;
            yPos = r * tileHeight + 10;

            visImg = insertText(visImg, [xPos yPos], labels{idx}, ...
                'FontSize', 18, 'BoxColor', 'black', 'TextColor', 'white', 'BoxOpacity', 0.6);
        end
    end
end
