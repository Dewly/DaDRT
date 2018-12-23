function [imgs, labels] = train_data_augmentation(img, label)

    imgs_c = cell(1,1); 
    labels_c = cell(1,1);
    idx = 0;
%-------------------------------------------------------------------------%    
    rotation_angle = [-45, -30, -20, -10, 10, 20, 30, 45];
%     rotation_angle = -45:5:45;
    for i = 1:numel(rotation_angle)
        im1 = imrotate(img, rotation_angle(i), 'bicubic', 'crop');
        label1 = imrotate(label, rotation_angle(i), 'bicubic', 'crop');
        idx = idx + 1;
        imgs_c{idx,1} = im1;
        labels_c{idx,1} = label1;
        %figure,imshow(mat2gray(img1))
    end
%-------------------------------------------------------------------------%
    g_sigma = [10, 7, 5, 3, 1, 0.7, 0.4, 0.1];
%     g_sigma = [10, 8, 6, 4, 2, 1, 0.8, 0.6, 0.4, 0.2, 0.1];
    for i =1:numel(g_sigma)
        w = fspecial('gaussian',[5 5],g_sigma(i));
        im2 = imfilter(img,w);
        idx = idx + 1;
        imgs_c{idx,1} = im2;
        labels_c{idx,1} = label;
        %figure,imshow(mat2gray(im2))
    end
%-------------------------------------------------------------------------%
    idx = idx + 1; imgs_c{idx,1} = fliplr(img); labels_c{idx,1} = label;
    idx = idx + 1; imgs_c{idx,1} = flipud(img); labels_c{idx,1} = label;
    idx = idx + 1; imgs_c{idx,1} = rot90(img,2); labels_c{idx,1} = label;
    idx = idx + 1; imgs_c{idx,1} = img; labels_c{idx,1} = label;
    %figure,imshow(mat2gray(im3))
%-------------------------------------------------------------------------%    
    imgs = cat(4,imgs_c{:});
    labels = cat(4,labels_c{:});
%-------------------------------------------------------------------------%    
end
