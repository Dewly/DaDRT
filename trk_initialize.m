function [tracker] = trk_initialize( img, region, gpu_idx)
% tracker = rnt_initialize( img, region )
%-------------------------------------------------------------------------%
    if (size(img, 3)==1), img = repmat(img, [1 1 3]); end
%-------------------------------------------------------------------------%    
% If the provided region is a polygon ...
    if numel(region) > 4
        x1 = min(region(1:2:end));
        x2 = max(region(1:2:end));
        y1 = min(region(2:2:end));
        y2 = max(region(2:2:end));
        region = [x1, y1, x2 - x1, y2 - y1];
    end 
%-region [x y w h]--------------------------------------------------------%
    tar_sz = region([4,3]);
    tar_ps = region([2,1]) + region([4,3])/2;
%-global params-----------------------------------------------------------%
    config.gpus = gpu_idx;
    config.tar_sz_mx = 80;
    config.window_sz_mx = 400;
    config.window_sz_mn = 200;
    config.output_sigma_factor = 0.08;
	config.tar_sz_c = tar_sz;
    config.tar_ps_c = tar_ps;
    config.scales = [0.95 1 1.05];
    config.randomSeed = 0;
    config.scaleLR = 0.6;  
    ratio_hw = tar_sz(1)/tar_sz(2);
    ratio_wh = 1/ratio_hw;
    p1 = 5;
    p2 = 9;
    if ratio_hw > 3
        p1 = 5;
        p2 = 13;   
    elseif ratio_hw > 2
        p1 = 5;
        p2 = 11; 
    end
    if ratio_wh > 3
        p1 = 13;
        p2 = 5;   
    elseif ratio_wh > 2
        p1 = 11;
        p2 = 5;
    end
    config.padding = [p1,p2];
%-------------------------------------------------------------------------%
    tar_scale = 1;
    win_scale = 1;
    if prod(tar_sz) > config.tar_sz_mx^2
        tar_scale = config.tar_sz_mx/max(tar_sz);
    end
    window_sz = tar_sz.*tar_scale.*config.padding;
    if prod(window_sz) < config.window_sz_mn^2
        win_scale = sqrt(config.window_sz_mn^2 / prod(window_sz));
    elseif prod(window_sz) > config.window_sz_mx^2
        win_scale = sqrt(config.window_sz_mx^2 / prod(window_sz));
    end
    window_sz = round(window_sz .* win_scale);
    config.sample_sz = round(tar_sz.*config.padding);
    config.init_wind_sz = window_sz;
    config.scl_tar = config.sample_sz ./ config.init_wind_sz;
%-------------------------------------------------------------------------%    
	rng(config.randomSeed);
%---init feature bone-----------------------------------------------------%
    prepareGPUs(config.gpus, true);
    [net_feat, net_feat_chs] = get_feature_net();
    net_feat.move('gpu');
    config.net_feat_in = net_feat.getInputs{1};
    config.net_feat_out = net_feat.getOutputs();
    config.net_feat_idx = net_feat.getVarIndex(config.net_feat_out);    
    rfs = net_feat.getVarReceptiveFields(config.net_feat_in);
    stride = [rfs(config.net_feat_idx).stride];
    config.stride = stride(1:2:end);
    sizes = net_feat.getVarSizes({config.net_feat_in, [config.init_wind_sz.*[1 1] 3 1]});
    feat_sz = {sizes{config.net_feat_idx}}; feat_sz_t = cell2mat(feat_sz');
    config.feat_sz = feat_sz_t(:,1:2);
    config.cf_idx = 1;
    config.net_img_avg = gpuArray( net_feat.meta.normalization.averageImage );
	config.img_avg = net_feat.meta.normalization.averageImage;
%---init regress net------------------------------------------------------%
    filter_sz = (tar_sz ./ config.scl_tar) ./ config.stride(1);  
    net_rgr = init_deep_regress_net([], config.net_feat_out, net_feat_chs, 48, filter_sz);
%-------------------------------------------------------------------------%    
    ft_in = [config.net_feat_out; feat_sz]; ft_in = ft_in(:)';
    rgr_sizes = net_rgr.getVarSizes(ft_in);
    %---------------------------------------------------------------------%
    config.idx_score = net_rgr.getVarIndex('feat_cf');
    net_rgr.vars(config.idx_score).precious = 1;
    config.rsp_sz = rgr_sizes{config.idx_score}(1:2);
%-------------------------------------------------------------------------%    
    motion_sigma_factor = 1.0;
    motion_sigma = sqrt(prod(filter_sz)) * motion_sigma_factor;    
    config.motion_map = gaussian_shaped_labels(motion_sigma, config.rsp_sz);
%-------------------------------------------------------------------------%    
    [crop, ~] = get_subwindow(img, tar_ps, config.sample_sz, config.init_wind_sz);
	label_inputs = get_label_inputs_simple(filter_sz, config.rsp_sz, config.output_sigma_factor);
    [crops, labels] = train_data_augmentation(crop, label_inputs{2});
    ns = size(crops,4);
    if ~isa(crops, 'gpuArray'), crops = gpuArray(crops); end
    crops = bsxfun(@minus, crops, config.net_img_avg);
%---prepare train data----------------------------------------------------%
    net_feat.eval({config.net_feat_in, crops});
    feat_train = {net_feat.vars(config.net_feat_idx).value};
    loss_wm = gpuArray( ones(1,1,1,ns,'single') );
    labels = gpuArray( labels );
    rgr_input_names = net_rgr.getInputs;
%-------------------------------------------------------------------------%
    config.train.ns = ns;
    config.train.feat = feat_train;
    config.train.label = labels;
    config.train.w = loss_wm;
%-------------------------------------------------------------------------%    
    config.update.cnt = 1;
    config.update.frq = 3;
    config.update.seed = config.randomSeed;
    config.update.feat = cellfun(@(x) repmat(x(:,:,:,end),[1,1,1,config.update.frq+1]),feat_train,'UniformOutput',false);
    config.update.label = repmat(labels(:,:,:,end),[1 1 1 config.update.frq+1]);
    config.update.w = ones(1,1,1,config.update.frq+1,'single');
    config.update.w(:,:,:,1) = 4;
    config.update.names = rgr_input_names;
%-------------------------------------------------------------------------%      
    derOutputs = {'objective', 1};
    batch_fn = @(db, batch) get_batch(db, batch, feat_train, labels, loss_wm, rgr_input_names);
%-------------------------------------------------------------------------%
    imdb = struct();
    imdb.images = struct(); % we keep the images struct for consistency with cnn_train_dag (MatConvNet)
    imdb.id = 1:ns;
    imdb.images.set = uint8(ones(1, ns)); % 1 -> train
    opts.train.gpus = config.gpus;
    opts.train.derOutputs = derOutputs;
    opts.train.numEpochs = 30;
    opts.train.learningRate = 8e-5;
    opts.train.weightDecay = 1;
    opts.train.batchSize = 1; % we empirically observed that small batches work better
    opts.train.profile = false;
    opts.train.randomSeed = config.randomSeed;
    opts.train.plotStatistics = false; % true false
    opts.train.print = false;
    opts.train.shuffle = true;
    opts.train.expDir = './data/train'; 
    cnn_train_dag_xc(net_rgr, imdb, batch_fn, opts.train);
    net_rgr.move('gpu');
%-------------------------------------------------------------------------%
    tracker.name = 'trk_dadrt';
    tracker.net_feat = net_feat;
    tracker.net_rgr = net_rgr;
    tracker.config = config;
%-------------------------------------------------------------------------%        
end


function inputs = get_batch(db, batch, feat_train, labels, loss_wm, rgr_input_names)
    batch_feat = cellfun(@(x) x(:,:,:,batch), feat_train, 'UniformOutput', false); 
%     batch_feat = cellfun(@gpuArray, batch_feat, 'UniformOutput', false); 
    batch_label = {labels(:,:,:,batch)};
    batch_w = {loss_wm(:,:,:,batch)};
    data_in = [batch_feat batch_label batch_w];
    input_ft = [ rgr_input_names; data_in];
    inputs = input_ft(:)';
end
