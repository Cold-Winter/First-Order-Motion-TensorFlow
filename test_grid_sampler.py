import torch
import tensorflow as tf
import tensorflow_addons as tfa

input_tensor = torch.arange(4*4).view(1, 1, 4, 4).float()
d = torch.linspace(-1, 1, 8)
meshx, meshy = torch.meshgrid((d, d))
grid = torch.stack((meshy, meshx), 2)
grid = grid.unsqueeze(0) # add batch dim
print(input_tensor.shape)

output = torch.nn.functional.grid_sample(input_tensor, grid, padding_mode='zeros', align_corners = False)
print(output)

output = torch.nn.functional.grid_sample(input_tensor, grid, padding_mode='border', align_corners = False)
print(output)




def grid_sampler_unnormalize(coord, side, align_corners):
    if align_corners:
        return ((coord + 1) / 2) * (side - 1)
    else:
        return ((coord + 1) * side - 1) / 2
        
def grid_sampler_compute_source_index(coord, size, align_corners):
    coord = grid_sampler_unnormalize(coord, size, align_corners)
    return coord

def safe_get(image, n, c, x, y, H, W):
    value = torch.Tensor([0])
    if  x >= 0 and x < W and y >=0 and y < H:
        value = image[n, c, y, x]
    return value

    
def bilinear_interpolate_torch_2D(image, grid, align_corners=False):
    '''
         input shape = [N, C, H, W]
         grid_shape  = [N, H, W, 2]
    
         output shape = [N, C, H, W]
    '''

    N, C, H, W = image.shape
    grid_H = grid.shape[1]
    grid_W = grid.shape[2]
    
    output_tensor = torch.zeros((N, C, grid_H, grid_W))
    for n in range(N):
        for w in range(grid_W):
            for h in range(grid_H):
                #get corresponding grid x and y
                x = grid[n, h, w, 1]
                y = grid[n, h, w, 0]
                
                #Unnormalize with align_corners condition
                ix = grid_sampler_compute_source_index(x, W, align_corners)
                iy = grid_sampler_compute_source_index(y, H, align_corners)
                
                x0 = torch.floor(ix).type(torch.LongTensor)
                x1 = x0 + 1

                y0 = torch.floor(iy).type(torch.LongTensor)
                y1 = y0 + 1
    
                #Get W matrix before I matrix, as I matrix requires Channel information
                wa = (x1.type(torch.FloatTensor)-ix) * (y1.type(torch.FloatTensor)-iy) 
                wb = (x1.type(torch.FloatTensor)-ix) * (iy-y0.type(torch.FloatTensor)) 
                wc = (ix-x0.type(torch.FloatTensor)) * (y1.type(torch.FloatTensor)-iy) 
                wd = (ix-x0.type(torch.FloatTensor)) * (iy-y0.type(torch.FloatTensor)) 
                
                #Get values of the image by provided x0,y0,x1,y1 by channel
                for c in range(C):
                    #image, n, c, x, y, H, W
                    Ia = safe_get(image, n, c, y0, x0, H, W)
                    Ib = safe_get(image, n, c, y1, x0, H, W)
                    Ic = safe_get(image, n, c, y0, x1, H, W)
                    Id = safe_get(image, n, c, y1, x1, H, W)
                    out_ch_val = torch.t((torch.t(Ia)*wa)) + torch.t(torch.t(Ib)*wb) + \
                                          torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)

                    output_tensor[n, c, h, w] = out_ch_val
    return output_tensor


output = bilinear_interpolate_torch_2D(input_tensor, grid, align_corners = True)
print(output)


















###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
##########################tensorflow implementation########################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

# Reference:
# [Yandong To Do] correct the align_corner part
# https://github.com/kevinzakka/spatial-transformer-network/blob/375f99046383316b18edfb5c575dc390c4ee3193/stn/transformer.py#L66
# https://github.com/OrkhanHI/pytorch_grid_sample_python/blob/main/pytorch_grid_sample_python.md
# https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

# [Yandong To Do] need to be fixed for align_corner
def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """

    # align corner = False
    # ((coord + 1) * side - 1) / 2

    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    # x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    # y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # modified by yandong for align corner (still not correct)
    x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32') - 1)
    y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32') - 1)

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out


tf_input_tensor = input_tensor.numpy().transpose((0,2,3,1))

out_sampler = bilinear_sampler(tf_input_tensor, grid[:,:,:,0], grid[:,:,:,1])
print(tf.transpose(out_sampler, perm=[0,3,1,2]))

batch_size, height, width, _ = tf_input_tensor.shape
# new_max = width - 1.0
# new_min = 0
# grid = (grid + 1.0) / 2.0 * (new_max - new_min)
grid = 0.5 * ((grid + 1.0) * tf.cast(height, 'float32') - 1)

out_sampler_add_on = tfa.image.resampler(tf_input_tensor, grid)
print(tf.transpose(out_sampler_add_on, perm=[0,3,1,2]))





