classdef ObjL2wLoss2 < dagnn.Loss
  properties
    nds = 6;
    wk = 1;
    ex = 0.5;
    wmask = [];
  end

  methods
	function outputs = forward(obj, inputs, params)%inputs H*W*C*B
        assert(numel(inputs)==3);
        assert(size(inputs{1},4)==1);
        r = inputs{1};
        y = inputs{2};
        wL = inputs{3};
        %-----------------------------------------------------------------%
        mr0 = exp(1.6.*(y-1)).*y;
        mr = mr0;
        rn = normalize_maxmin(abs(r));
        [~,pt] = localmax_nonmaxsup2d(rn);
        if ~isempty(pt)
            for tk = 1:obj.nds
                dv  = ceil(pt(tk,1)-(size(r,1)/2+0.5));
                dh  = ceil(pt(tk,2)-(size(r,2)/2+0.5)); 
                if abs(dh)<3&&abs(dv)<3
                    yd = obj.ex.*mr0;
                else
                    yd = circshift(mr0,[dv dh]);
                end
                mr = mr + yd;
            end
        end
        obj.wmask = mr;
        %-----------------------------------------------------------------%
        loss_ry = obj.wmask .* (r - y).^2;
        loss_ry_w = bsxfun(@times, loss_ry, wL);
        outputs{1} = sum( loss_ry_w(:) );
        %-----------------------------------------------------------------%
        n = obj.numAveraged ;
        m = n + size(inputs{1},4) ;
        obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
        obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        r = inputs{1};
        y = inputs{2};
        wL = inputs{3};
        n = 1;
%-----------------------------------------------------------------%
        dr = (derOutputs{1}/n) .* obj.wmask .* 2 .* (r-y);
        derInputs{1} = bsxfun(@times, dr, wL);
        derInputs{2} = {};
        derInputs{3} = {};
        derParams = {} ;
    end

    function obj = ObjL2wLoss2(varargin)
      obj.load(varargin) ;
    end
  end
end
