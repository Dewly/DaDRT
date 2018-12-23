function [net, feat_ch] = get_feature_net()
%-------------------------------------------------------------------------%
   vgg16 = load('imagenet-vgg-verydeep-16.mat');
    nlayers = vgg16.layers(1:30);
    nlayers(24) = []; % [17 24] remove last two pooling layer
    nlayers(17) = [];
    avgImg = vgg16.meta.normalization.averageImage; % matlab->rgb; caffe->bgr
    net.layers = nlayers;
    net.meta.normalization.averageImage = avgImg;
    net = vl_simplenn_tidy(net) ;
    net = dagnn.DagNN.fromSimpleNN(net);
    featIdx = [17, 23, 29];
    [net.vars(featIdx).fanout] = deal(0);
    feat_ch = [256, 512, 512];
    clear vgg16;
%-------------------------------------------------------------------------%    
end
