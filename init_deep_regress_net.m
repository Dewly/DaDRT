function [net] = init_deep_regress_net(bone, inputs, chs_in, chs_ad, filter_sz)
%-------------------------------------------------------------------------%    
    cudnnWorkspaceLimit = 1024*1024*1024 * 2; 
%-------------------------------------------------------------------------%
    if isempty(bone)
        net = dagnn.DagNN();  
    else
        net = bone.copy();
    end
%-------------------------------------------------------------------------%
    ck = numel(inputs);
    ft = cell(1,ck);
    for k = 1:ck
        [net,ft{k}] = add_feature_reinit_block(net, inputs{k}, chs_in(k), chs_ad, ['f' num2str(k) '_'], 1, 2);
    end
%-------------------------------------------------------------------------%
    layerName = 'feat'; layer_in = ft; feat = layerName;
    net.addLayer(layerName, dagnn.Concat(), layer_in, feat) ;
%-------------------------------------------------------------------------%
    fh = ceil(filter_sz(1));
    fw = ceil(filter_sz(2));
    %-----------------------------------------------------------------%
    layerName =  'feat_cf'; layer_in = feat; layer_out = layerName;
    pars = {[layerName 'f'], [layerName 'b']}; 
    layer = dagnn.Conv('size',[fh,fw,chs_ad*ck,1],'pad',0,'hasBias',true,'opts',{'cudnnworkspacelimit',cudnnWorkspaceLimit});  
    net.addLayer(layerName, layer, layer_in, layer_out, pars);  
    %-----------------------------------------------------------------%
    f = net.getParamIndex(pars{1}) ;
    net.params(f).value = single(sqrt(2)*randn(fh,fw,chs_ad*ck,1)/sqrt(fh*fw*chs_ad*ck))/1e2;
    net.params(f).learningRate = 1;
    net.params(f).weightDecay = 1;
    %---------------------------------------------------------------------%
    f = net.getParamIndex(pars{2}) ;
    net.params(f).value = single(zeros(1,1));
    net.params(f).learningRate = 2;
    net.params(f).weightDecay = 1;
%---loss layers-----------------------------------------------------------%
    net.addLayer('objective', dagnn.ObjL2wLoss2(), {layer_out, 'glabel', 'loss_w'}, 'objective');
%-------------------------------------------------------------------------%
end
