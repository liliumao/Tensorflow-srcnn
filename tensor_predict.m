clc;
clear;

up_scale = 2;
size_gnd = [1080,1920];
HD = true;

if(strcmp(model_type, '9-1-5'))
    size_clip = [6,6];
else if(strcmp(model_type, '9-5-5'))
        size_clip = [8,8];
    end
end

folder = 'test';
filepaths = '1.bmp';
outfilepaths = 'out.bmp';

im = imread(fullfile(folder,filepaths));
if HD
    if size(im, 3)>1
        im_gnd = rgb2ycbcr(im);
        im_cb = im_gnd(:,:,2);
        im_cr = im_gnd(:,:,3);
        im_gnd = im_gnd(:,:,1);
    end
    im_gnd = modcrop(im_gnd, up_scale);
    im_l = imresize(im_gnd, 1/up_scale, 'bicubic');

else
    if size(im, 3)>1
        im = rgb2ycbcr(im);
        im_cb = imresize(im(:,:,2), size_gnd);
        im_cr = imresize(im(:,:,3), size_gnd);
        im_l = im(:,:,1);
    end
end


im_b = imresize(im_l, size_gnd, 'bicubic');
im_l = modcrop(im_b, up_scale);
im_l = single(im_l)/255;

im_h_clip = imread(outfilepaths);

im_h_clip = uint8(im_h_clip*255);
im_h = im_b;
im_h(size_clip(1)+1:end-size_clip(1), size_clip(2)+1:end-size_clip(2)) = im_h_clip;

if HD
    psnr_bicubic = compute_psnr(double(im_gnd(size_clip(1)+1:end-size_clip(1), size_clip(2)+1:end-size_clip(2))),...
        double(im_b(size_clip(1)+1:end-size_clip(1), size_clip(2)+1:end-size_clip(2))));
    psnr_srcnn = compute_psnr(im_gnd(size_clip(1)+1:end-size_clip(1), size_clip(2)+1:end-size_clip(2)), ...
        im_h(size_clip(1)+1:end-size_clip(1), size_clip(2)+1:end-size_clip(2)));

    fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bicubic);
    fprintf('PSNR for SRCNN Reconstruction: %f dB\n', psnr_srcnn);
end

im_h_ycbcr = zeros([size_gnd(1), size_gnd(2), 3]);
im_h_ycbcr(:,:,1) = im_h;
im_h_ycbcr(:,:,2) = im_cb;
im_h_ycbcr(:,:,3) = im_cr;

im_h = ycbcr2rgb(uint8(im_h_ycbcr));
