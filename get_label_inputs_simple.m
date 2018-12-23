function inputs = get_label_inputs_simple(obj_sz, resp_sz, sig_ratio)
% GET_LABEL_INPUTS_SIMPME returns the network inputs that specify the labels.
%-------------------------------------------------------------------------%
    sigmas = obj_sz' .* sig_ratio;
    glabel = gaussian_shaped_labels(sigmas, resp_sz);
    glabel = permute(glabel,[1 2 4 3]);
%-------------------------------------------------------------------------%    
    inputs = {'glabel', single(glabel)};
%-------------------------------------------------------------------------%
end
