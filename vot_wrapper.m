function vot_wrapper
% rnt VOT integration 

% *************************************************************
% VOT: Always call exit command at the end to terminate Matlab!
% *************************************************************
    cleanup = onCleanup(@() exit() );
% *************************************************************    
    addpath(genpath('/mnt/duming/matconvnet25/matlab'));
    addpath(genpath('/mnt/duming/matconvnet25/contrib/autonn'));
    addpath(genpath('/mnt/duming/matconvnet25/contrib/mcnExtraLayers'));
    vl_setupnn;
    setup_autonn;%vl_contrib('setup','autonn')
    setup_mcnExtraLayers; %vl_contrib('setup','mcnExtraLayers')
% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************
    RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock))); 
% **********************************
% VOT: Get initialization data
% **********************************
    [handle, image, region] = vot('rectangle');    
% Initialize the tracker
    gpu_idx = 2;
    trk_tracker = trk_initialize(single(imread(image)), region, gpu_idx);   
	while true
        % **********************************
        % VOT: Get next frame
        % **********************************
        [handle, image] = handle.frame(handle);
        if isempty(image)
            break;
        end
        % Perform a tracking step, obtain new region
        [trk_tracker, region] = trk_tracker_update(trk_tracker, single(imread(image)));
        % **********************************
        % VOT: Report position for frame
        % **********************************
        handle = handle.report(handle, double(region));
	end
% **********************************
% VOT: Output the results
% **********************************
    handle.quit(handle);
end
