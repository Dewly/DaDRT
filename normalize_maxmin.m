function [map] = normalize_maxmin(r)
    ma = max(max(r,[],1),[],2);
	mi = min(min(r,[],1),[],2);
	map = (r-mi)./(ma-mi+eps);
end
