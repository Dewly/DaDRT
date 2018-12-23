function [label] = gaussian_shaped_labels(sigma, sz)
%GAUSSIAN_SHAPED_LABELS
%   Gaussian-shaped labels for all shifts of a sample.
%
%   LABELS = GAUSSIAN_SHAPED_LABELS(SIGMA, SZ)
%   Creates an array of labels (regression targets) for all shifts of a
%   sample of dimensions SZ. The output will have size SZ, representing
%   one label for each possible shift. The labels will be Gaussian-shaped,
%   with the peak at 0-shift (top-left element of the array), decaying
%   as the distance increases, and wrapping around at the borders.
%   The Gaussian function has spatial bandwidth SIGMA.
    if isscalar(sigma)
        sigma = [sigma sigma];
    end    
    [rs, cs] = ndgrid((0.5:sz(1)-0.5) - (sz(1)/2), (0.5:sz(2)-0.5) - (sz(2)/2));
    gs = exp(-0.5* (((rs.^2/sigma(1)^2 + cs.^2/sigma(2)^2) ))); 
    label = normalize_maxmin(gs);
end

