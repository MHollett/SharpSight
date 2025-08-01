function processAdaptivePipeline(inputFolder, outputFolder)
% Calls our experimental code

    processDroneDataset(inputFolder, outputFolder);
    
    fprintf('Adaptive pipeline complete using SharpSight preprocessing pipeline\n');
end