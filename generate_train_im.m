clear;close all;
%% settings
folder = 'Train';
savepath = 'traintest.h5';
size_input = 33;
size_label = 17;
scale = 2;
stride = 14;
size_output = 21888;

%% initialization
data = zeros(1, size_input, size_input, 1);
label = zeros(1, size_label, size_label, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.JPEG'));
    
file_order = randperm(length(filepaths));

for i = 1 : length(filepaths)
    image = imread(fullfile(folder,filepaths(file_order(i)).name));
    if numel(size(image)) < 3
        continue;
    end
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
    
    im_label = modcrop(image, scale);
    [hei,wid] = size(im_label);
    im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);

            if rand(1) >= 0.5
                count=count+1
                data(count, :, :, 1) = subim_input;
                label(count, :, :, 1) = subim_label;
            end
            
            if count == size_output
                break;
            end
        end
        
        if count == size_output
            break;
        end
    end
    if count == size_output
        break;
    end
end

order = randperm(count);
data = data(order, :, :, 1);
label = label(order, :, :, 1); 

count

%% writing to HDF5
chunksz = 128;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(last_read+1:last_read+chunksz,:,:,1); 
    batchlabs = label(last_read+1:last_read+chunksz,:,:,1);

    startloc = struct('dat',[totalct+1,1,1,1], 'lab', [totalct+1,1,1,1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
