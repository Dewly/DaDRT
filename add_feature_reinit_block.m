function [net,blk_out] = add_feature_reinit_block(net, blk_in, ch_in, ch_ad, prefix, lrt, scl)
%-------------------------------------------------------------------------%
    if nargin < 6, lrt = 1; scl = 1; end
    if nargin < 7, scl = 1; end
%-------------------------------------------------------------------------%
	cudnnWorkspaceLimit = 1024*1024*1024 * 2; 
%-------------------------------------------------------------------------%
    layerName = [prefix 'cv_1x1']; layer_in = blk_in; layer_out = layerName;
	pars = {[layerName 'f'], [layerName 'b']}; 
	layer = dagnn.Conv('size', [1,1,ch_in,ch_ad], 'hasBias', true, 'opts', {'cudnnworkspacelimit', cudnnWorkspaceLimit});  
	net.addLayer(layerName, layer, layer_in, layer_out, pars);        
    %---------------------------------------------------------------------%
    f = net.getParamIndex(pars{1}) ;
    net.params(f).value = single(sqrt(2)*randn(1,1,ch_in,ch_ad)/sqrt(1*1*ch_in));
    net.params(f).learningRate = lrt;
    net.params(f).weightDecay = 1;
    %---------------------------------------------------------------------%
    f = net.getParamIndex(pars{2}) ;
    net.params(f).value = single(zeros(ch_ad,1));
    net.params(f).learningRate = 2;
    net.params(f).weightDecay = 1; 
%-------------------------------------------------------------------------%
    layerName =[prefix 'nm']; layer_in = layer_out; layer_out = layerName;
	pars = {[layerName '_w']};
	layer = dagnn.Normalize();
	net.addLayer(layerName, layer, layer_in, layer_out, pars);
    %---------------------------------------------------------------------%
    f = net.getParamIndex(pars{1}) ;
    net.params(f).value = single(rand(1,1,ch_ad));
    net.params(f).learningRate = scl;
    net.params(f).weightDecay = 1;
%-------------------------------------------------------------------------%    
    layerName =  [prefix 'ru']; layer_in = layer_out; layer_out = layerName;
    net.addLayer(layerName, dagnn.ReLU(), layer_in, layer_out) ;
%-------------------------------------------------------------------------%
    blk_out = layer_out;
%-------------------------------------------------------------------------%
end
