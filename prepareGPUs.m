function prepareGPUs(idxs, cold)
%prepareGPUs(opts, cold)
%-------------------------------------------------------------------------%
	numGpus = numel(idxs) ;
    if numGpus > 1
      %-check parallel pool integrity as it could have timed out----------%
        pool = gcp('nocreate') ;
        if ~isempty(pool) && pool.NumWorkers ~= numGpus
            delete(pool) ;
        end
        pool = gcp('nocreate') ;
        if isempty(pool)
            parpool('local', numGpus) ;
            cold = true ;
        end
    end
	if numGpus >= 1 && cold
%         fprintf('%s: resetting GPU\n', mfilename)
        clear vl_tflow vl_imreadjpeg ;       
        if numGpus == 1
            gpuDevice(idxs);
        else
            spmd
            clearMex() ;
            gpuDevice(idxs(labindex));
            end
        end
	end
%-------------------------------------------------------------------------%
end
