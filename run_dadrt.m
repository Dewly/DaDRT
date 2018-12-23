function results=run_dadrt(seq, res_path, bSaveImage)
% results=run_dadr(seq, res_path, bSaveImage)
    close all

    nimg = numel(seq.s_frames);
    im = single(imread(seq.s_frames{1}));
    region = seq.init_rect; 
    gpu_idx = 1;
    trk_tracker = trk_initialize(im, region, gpu_idx); %region->[x y w h]
    
    time = 0;  %to calculate FPS
    positions = zeros(nimg, 2);  %to calculate precision
    rect_position = zeros(nimg, 4);

    for frame = 1:nimg
        tic()
        if frame > 1
            imc = single(imread(seq.s_frames{frame}));
            [trk_tracker, region] = trk_tracker_update(trk_tracker, imc);
        end
        positions(frame,:) = trk_tracker.config.tar_ps_c;
        rect_position(frame, :) = region;          
		time = time + toc();

        if bSaveImage
            if frame == 1  %first frame, create GUI
    %             figure('Number','off', 'Name',['Tracker - ' video_path])
                im_handle = imshow(im, 'Border','tight', 'InitialMag',200);
                rect_handle = rectangle('Position',rect_position(frame,:), 'EdgeColor','g');
            else
                try  %subsequent frames, update GUI
                    set(im_handle, 'CData', im)
                    set(rect_handle, 'Position', rect_position(frame,:))
                catch  %#ok, user has closed the window
                    return
                end
            end
            imwrite(frame2im(getframe(gcf)), fullfile(res_path,[num2str(frame) '.jpg'])); 
        end
        drawnow % 	pause(0.05)  %uncomment to run slower
    end
    
    fps = nimg / time;
    disp(['fps: ' num2str(fps)])
    results.type = 'rect';
    results.res = rect_position;%each row is a rectangle
    results.fps = fps;
end
