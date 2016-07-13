import os
import sys
import cv2
import math
import numpy as np
import caffe
import skimage


def preprocess_frame(image, target_height=224, target_width=224):

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))


def get_features(net, transformer, image_list, layers='fc7', layer_sizes=[4096], batch_size=10):
    iter_until = len(image_list) + batch_size
    all_feats = np.zeros([len(image_list)] + layer_sizes)

    for start, end in zip(range(0, iter_until, batch_size), \
                          range(batch_size, iter_until, batch_size)):

        image_batch = image_list[start:end]

        caffe_in = np.zeros(np.array(image_batch.shape)[[0,3,1,2]], dtype=np.float32)
        
        for idx, in_ in enumerate(image_batch):
            caffe_in[idx] = transformer.preprocess('data', in_)

        out = net.forward_all(blobs=[layers], **{'data':caffe_in})
        feats = out[layers]

        all_feats[start:end] = feats

    return all_feats


def extract_feature_from_video_frames( output_file, frame_list, videoName ):
    model_file = '../../models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel'
    deploy_prototxt = '../../models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers_deploy.prototxt'
#     model_file = '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
#     deploy_prototxt = '../../models/bvlc_reference_caffenet/deploy.prototxt'
    net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)
    layer = 'fc7'
    if layer not in net.blobs:
        raise TypeError("Invalid layer name: " + layer)
    
    imagemean_file = '../../python/caffe/imagenet/ilsvrc_2012_mean.npy'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(imagemean_file).mean(1).mean(1))
    transformer.set_raw_scale('data', 255.0)
    transformer.set_channel_swap('data', (2,1,0))
    
    cropped_frame_list = np.array(map(lambda x: preprocess_frame(x), frame_list))
    all_feats = get_features(net, transformer, cropped_frame_list )

    for i in range(len(all_feats)):
        with open(output_file, 'a') as f:
            f.write( '{}_frame_{},'.format( videoName,i+1) )
            np.savetxt(f, all_feats[None,i], fmt='%g', delimiter=',')
    
    return all_feats


templete_path = './results/val.s2vt_vgg_rgb_beam_size_1_all.txt'
video_path = '/data/public-dataset/YouTubeClips/val'
#video_path = './video_source'

videos = []
with open(templete_path) as file :
    for line in file :
        temp = line.split()
        videos.append(temp[1]+'.avi')
        
#videos = os.listdir(video_path)
#videos = filter(lambda x: x.endswith('avi'), videos)

output_file = './MSVD_val_vgg19.txt' 
with open(output_file, 'w') as f:
    f.write('')

for video in videos:
    print video

    frame_count = 0
    frame_list = []
    video_fullpath = os.path.join(video_path, video)
    
    try:
        cap  = cv2.VideoCapture( video_fullpath )
    except:
        pass


    while True:
        retval, frame = cap.read()
        # image is a numpy array containing the next frame
        # do something with image here
        if not retval:
            # Message to be displayed after releasing the device
            print "Released Video Resource"
            break

        frame_list.append(frame)
        frame_count += 1        

    num_frames = int(math.ceil(frame_count/10.0))
    print("output_frame:{}".format(num_frames))
    frame_list_np = np.array(frame_list)
    # sample video frames over a interval 
    if frame_count > num_frames:
        frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
        frame_list_np = frame_list_np[frame_indices]
   
    videoName = os.path.splitext(video)[0]
    all_feats = extract_feature_from_video_frames( output_file, frame_list_np, videoName )
                
print('Feature extraction is done!')       