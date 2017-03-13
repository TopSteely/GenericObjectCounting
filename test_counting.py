# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
#from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
from datasets.factory_counting import get_imdb
import datasets.imdb
from roi_data_layer.minibatch_counting import get_minibatch
from fast_rcnn.train_counting import get_training_roidb
from scipy.misc import imread

def combined_roidb(imdb_names):
    
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for testing '.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb
    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]

    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im = im.astype(np.float32, copy=True)
    im -= cfg.PIXEL_MEANS

    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
	# TODO: this is needed now to get the train and test the same.
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes_in, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in 1: # xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, max_per_image=400, thresh=-np.inf, vis=False):
    """Test a Fast R-CNN network on an image database."""

    imdb, roidb = combined_roidb(cfg.TEST.BOXES_FILE)
    output_dir = get_output_dir(imdb, net)
    print '{:d} roidb entries'.format(len(roidb))
    roidb = roidb[0]
    #print net.params['iep'][0].data
    
    # print "conv:", net.params['conv_new_1'][0].data
    #roidb = roidb[0:1000]
    error = np.zeros(20)
    max_patches = 0
    for image_input in roidb:
	# TODO:
	#if not image_input['image']=='/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/001825.jpg':
	#	continue
	#if not image_input['image']=='/var/scratch/spintea/Repositories/ms-caffe/data/VOCdevkit2007/VOC2007/JPEGImages/004554.jpg':
	#	continue
	print "input image: ", image_input['image']
        im = cv2.imread(image_input['image'])
        data_blob, im_scale_factors  = _get_blobs(im, image_input['boxes'])

        # overlaps (227, 21, 1, 1)  -> (674, 21, 1, 1)
        # rois (227, 5, 1, 1)       -> (674, 5, 1, 1)
        # functions (227, 2, 1, 1)  -> (674, 2, 1, 1)
        # labels (1, 21, 1, 1)      -> (1, 21, 1, 1)
        # data (1, 3, 600, 901)     -> (1, 3, 850, 600)
        image_input['labels']=image_input['labels'][:,1:]

        #batch_ind = np.ones((image_input['boxes'].shape[0], 1))
        #if image_input['boxes'].shape[0] > max_patches:
        #    max_patches = image_input['boxes'].shape[0]
        #image_input['boxes'] = np.hstack((batch_ind, image_input['boxes']))

	image_input['boxes'] = _get_rois_blob(image_input['boxes'], im_scale_factors)

        image_input['boxes'] = image_input['boxes'].reshape(image_input['boxes'].shape[0], image_input['boxes'].shape[1], 1, 1)
        image_input['labels'] = image_input['labels'].reshape(image_input['labels'].shape[0], image_input['labels'].shape[1], 1, 1)
        image_input['functions'] = image_input['functions'].reshape(image_input['functions'].shape[0], image_input['functions'].shape[1], 1, 1)
        image_input['overlaps'] = image_input['overlaps'].reshape(image_input['overlaps'].shape[0], image_input['overlaps'].shape[1], 1, 1)
        scores = net_predict(image_input, data_blob, net)
        #print scores['iep'][:,:], image_input['labels'][0,:,0,0]
        #raw_input()
        pred = np.mean(scores['iep'][:,:],axis=0)
        labels = image_input['labels'][0,:,0,0]
        error += np.abs(pred - labels)
    print 'evaluated', len(roidb)
    print 'class error: ', error/float(len(roidb))
    print 'Mean error: ', np.mean(error/float(len(roidb)))


def net_predict(image_input, data_blob, net):
        net.blobs['data'].reshape(*(data_blob['data'].shape))
        net.blobs['rois'].reshape(*(image_input['boxes'].shape))
        #net.blobs['labels'].reshape(*(image_input['labels'].shape))
        net.blobs['functions'].reshape(*(image_input['functions'].shape))
        #net.blobs['overlaps'].reshape(*(image_input['overlaps'].shape))
	net.reshape()

        # do forward
        forward_kwargs = {'data': data_blob['data'].astype(np.float32, copy=False)}
        forward_kwargs['rois'] = image_input['boxes'].astype(np.float32, copy=False)
        #forward_kwargs['labels'] = image_input['labels'].astype(np.float32, copy=False)
        forward_kwargs['functions'] = image_input['functions'].astype(np.float32, copy=False)
        #forward_kwargs['overlaps'] = image_input['overlaps'].astype(np.float32, copy=False)
        scores = net.forward(**forward_kwargs)

	#print '>>> conv_new_1: ', net.blobs['data'].data
	#print '>>> conv_new_1: ', net.blobs['conv_new_1'].data
	#print '>>> rfcn_clss params:',net.params['rfcn_cls'][0].data
	#print '>>> rfcn: ', net.blobs['rfcn_cls'].data
	#print '>>> w-iep: ', net.params['iep'][0].data
	#print '>>> iep: ', net.blobs['iep'].data
	#print 'scores: ',scores
	return scores
