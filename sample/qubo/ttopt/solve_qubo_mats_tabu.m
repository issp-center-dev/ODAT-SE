matDir = 'output_qubo_mats';
matFiles = dir(fullfile(matDir, '*.mat'));

for k = 1:length(matFiles)
    matFilePath = fullfile(matDir, matFiles(k).name);
    data = load(matFilePath);
    qprob = qubo(data.q_mat);
    result = solve(qprob);
    outputDir = 'output_qubo_mats_sol';
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    fprintf('Results for %s:\n', matFiles(k).name);
    disp(result);
    
    outputFilePath = fullfile(outputDir, sprintf('%s_result.txt', matFiles(k).name(1:end-4)));
    fid = fopen(outputFilePath, 'w');
    fprintf(fid, 'Results for %s:\n', matFiles(k).name);
    fprintf(fid, '%s\n', mat2str(result.BestX));
    fprintf(fid, '%.15g\n', result.BestFunctionValue);
    fclose(fid);
end