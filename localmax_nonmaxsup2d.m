function [local_max, local_ps] = localmax_nonmaxsup2d(response)
    BW = imregionalmax(response);
    CC = bwconncomp(gather(BW));

    local_max = max(response(:));
    [h,w] = find(response == max(response(:)));
    local_ps = [mean(h) mean(w)];
    if length(CC.PixelIdxList) > 1
        temp_ps = cell2mat(CC.PixelIdxList');
        temp_mx = response(temp_ps);
        [local_max, idx] = sort(temp_mx, 'descend');
        temp_pss = temp_ps(idx);
        [psh,psw] = ind2sub(size(response),temp_pss);
        local_ps = [psh psw];
    end
end  % endfunction