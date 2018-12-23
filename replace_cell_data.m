function [x] = replace_cell_data(x,y,idx)
    x(:,:,:,idx)=y;
end