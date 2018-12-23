function [rnt, region] = trk_tracker_update(rnt, img)
%[rnt, region] = trk_tracker_update(rnt, img)
%-------------------------------------------------------------------------%
    config = rnt.config;
%-------------------------------------------------------------------------%
    if (size(img, 3)==1), img = repmat(img, [1 1 3]); end
%-------------------------------------------------------------------------%
    [crop, ~] = get_subwindow(img, config.tar_ps_c, config.sample_sz, config.init_wind_sz);
    if ~isa(crop, 'gpuArray'), crop = gpuArray(crop); end
    crop = bsxfun(@minus, crop, config.net_img_avg);
    rnt.net_feat.eval({config.net_feat_in, crop});
    feat = {rnt.net_feat.vars(config.net_feat_idx).value};
    input_ft = [config.net_feat_out; feat];
    input_ft = input_ft(:)';
    rnt.net_rgr.eval(input_ft);
    rsp_o = gather(rnt.net_rgr.vars(config.idx_score).value);
    rsp_r = imresize(rsp_o, config.stride(config.cf_idx), 'bicubic');
    m_r = imresize(config.motion_map, config.stride(config.cf_idx), 'bicubic');
    rsp = m_r .* rsp_r;
    pk = max(rsp(:));
    [vert_delta, horiz_delta] = find(rsp == pk);
    vert_delta = mean(vert_delta);
    horiz_delta = mean(horiz_delta);
    vert_delta  = vert_delta  - (size(rsp,1)/2+0.5);
    horiz_delta = horiz_delta - (size(rsp,2)/2+0.5);           
    config.tar_ps_c = config.tar_ps_c + config.scl_tar .* [vert_delta, horiz_delta];  
%-------------------------------------------------------------------------%
    if max(rsp(:)) > 0.1
        ns = length(config.scales);
        sps = gpuArray( single( zeros([config.init_wind_sz 3 ns]) ) );
        for i = 1:length(config.scales)
            sample_sz = round( config.tar_sz_c .* config.scales(i) .* config.padding );
            [crop, ~] = get_subwindow(img, config.tar_ps_c, sample_sz, config.init_wind_sz);
            if ~isa(crop, 'gpuArray'), crop = gpuArray(crop); end
            crop = bsxfun(@minus, crop, config.net_img_avg);
            sps(:,:,:,i) = single( crop ); 
        end
        rnt.net_feat.eval({config.net_feat_in, sps});
        feat_scl = {rnt.net_feat.vars(config.net_feat_idx).value};
        input_ft_scl = [config.net_feat_out; feat_scl];
        input_ft_scl = input_ft_scl(:)';
        rnt.net_rgr.eval(input_ft_scl); 
        rsp_scl = gather(rnt.net_rgr.vars(config.idx_score).value);
        rsp_scl = rsp_scl .* config.motion_map;
        rsp_scl = permute(rsp_scl,[1 2 4 3]);
        [~, ~, sid] = ind2sub(size(rsp_scl),find(rsp_scl == max(rsp_scl(:)),1));
        scl_cur = config.tar_sz_c * config.scales(sid);
        config.tar_sz_c  = (1-config.scaleLR)*config.tar_sz_c  + config.scaleLR*scl_cur;
        config.sample_sz = round( config.tar_sz_c .* config.padding );
        config.scl_tar = config.sample_sz ./ config.init_wind_sz;
    end 
%-------------------------------------------------------------------------% 
    [crop, ~] = get_subwindow(img, config.tar_ps_c, config.sample_sz, config.init_wind_sz);
    if ~isa(crop, 'gpuArray'), crop = gpuArray(crop); end
    crop = bsxfun(@minus, crop, config.net_img_avg);
    rnt.net_feat.eval({config.net_feat_in, crop});
    feat_update = {rnt.net_feat.vars(config.net_feat_idx).value};
    szz = (config.tar_sz_c / config.scl_tar) / config.stride(config.cf_idx);
    label_inputs = get_label_inputs_simple(szz, config.rsp_sz, config.output_sigma_factor);     
%-------------------------------------------------------------------------%
    config.update.cnt = config.update.cnt+1;
    if config.update.cnt>config.update.frq+1
        config.update.cnt = 2;
    end
    up_idx = cell(1,length(config.update.feat));
    up_idx(:) = deal({config.update.cnt});
    config.update.feat = cellfun(@replace_cell_data,config.update.feat,feat_update,up_idx,'UniformOutput',false);
    config.update.label(:,:,:,config.update.cnt) = label_inputs{2};  
%-------------------------------------------------------------------------%
    batch_fn = @(db, batch) get_batch(db, batch, config.update.feat, config.update.label, config.update.w, config.update.names);
    net_u = rnt.net_rgr;
    imdb = struct();
    imdb.images = struct(); % we keep the images struct for consistency with cnn_train_dag (MatConvNet)
    imdb.id = 1:(config.update.frq+1);
    imdb.images.set = uint8(ones(1, config.update.frq+1)); % 1 -> train
    opts.train.gpus = config.gpus;
    opts.train.derOutputs = {'objective', 1};
    opts.train.numEpochs = 2;
    opts.train.learningRate = 3e-5;
    opts.train.weightDecay = 1;
    opts.train.batchSize = 1; 
    opts.train.profile = false;
    opts.train.plotStatistics = false;
    opts.train.print = false;
    opts.train.shuffle = true;
    opts.train.randomSeed = config.update.seed;
    config.update.seed = config.update.seed + opts.train.numEpochs;
    opts.train.expDir = './data/update'; 
    cnn_train_dag_xc(net_u, imdb, batch_fn, opts.train);   
    net_u.move('gpu'); 
    rnt.net_rgr = net_u;
%-------------------------------------------------------------------------% 
    region = [config.tar_ps_c([2,1])-config.tar_sz_c([2,1])/2, config.tar_sz_c([2,1])];
    region = double(gather(region));
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
    rnt.config = config;
end

function inputs = get_batch(db, batch, feat_train, labels, loss_wm, rgr_input_names)
    batch_feat = cellfun(@(x) x(:,:,:,batch), feat_train, 'UniformOutput', false); 
    batch_label = {labels(:,:,:,batch)};
    batch_w = {loss_wm(:,:,:,batch)};
    data_in = [batch_feat batch_label batch_w];
    input_ft = [ rgr_input_names; data_in];
    inputs = input_ft(:)';
end


