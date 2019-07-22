# Faster R-CNN is the first object detector in the RCNN line that uses only deep learning
# RCNN and Fast RCNN still used the traditional 'Selective Search' Algorithm which was slow
# Faster R-CNN replaces the region proposal algorithm with a Region Proposal Network (RPN)

# We need to understand the following four topics in order to build Faster RCNN
# 1. Region Proposal Network (RPN), anchor boxes
# 2. RPN loss functions
# 3. Region of Interest Pooling (ROI pooling)
# 4. ROI loss functions


# ---------------------- FEATURE EXTRACTION ----------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Say this is our format of the Neural network
# 1. A 800x800 three channel image
# 2. Bounding boxes given in the format of [y1, x1, y2, x2]
# 3. With labels in integers, where 0 indicates background

image = torch.zeros((1, 3, 800, 800)).float()
bounding_box = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]])  # [y1, x1, y2, x2]
labels = torch.LongTensor([6, 8])
sub_sample = 16

# We will use the VGG16 network as our feature extraction module
# which will act as both the backbone for the RPN and the Fast-RCNN network

dummy_image = torch.zeros((1, 3, 800, 800)).float()
model = torchvision.models.vgg16(pretrained=True)
feature_extractor = list(model.features)

req_features = []
k = dummy_image.clone()
for i in feature_extractor:
    k = i(k)
    if k.size()[2] < (800//sub_sample):
        break
    req_features.append(i)
    out_channels = k.size()[1]

print(len(req_features))  # 30
print(out_channels)  # 512

faster_rcnn_feature_extractor = nn.Sequential(*req_features)

# Extracting Feature through VGG16
out_map = faster_rcnn_feature_extractor(image)
print(out_map.size())

# ---------------------- ANCHOR BOXES ----------------------
# Since we have pooled our image from 800 to 25, we have a sub sampling ratio of 32
# Every pixel in the output feature map maps to corresponding 32x32 pixels in the image (25*32 = 800)
# Because we used convolution, the spatial structure is also correct,
# as in the top-left pixel of the 25x25 output maps to the top left 32x32 of the input image

# We first generate anchor boxes on top of the 32x32 pixels first
# At each pixel location on the feature map, we generate 9 anchor boxes (3 ratio, 3 scales)
# Each anchor box will have [y1, x1, y2, x2]
# So at each pixel location of the feature map, an anchor will have the shape (9, 4)

# The above are the anchor locations at the first feature map pixel
# Since we got 9 anchors at 25x25 locations, we will get 25x25x9 = 5625 anchors in total

# Now generalizing the above to all of the feature map locations,
# We first generate centers for eah and every feature map pixel

# Initialize anchor boxes with zeros
anchor_ratio = torch.tensor([0.5, 1, 2]).type(torch.FloatTensor)
anchor_scale = torch.tensor([8, 16, 32]).type(torch.FloatTensor)
anchor_base = torch.zeros((len(anchor_ratio) * len(anchor_scale), 4), dtype=torch.float32)

feature_map_size = 800 // sub_sample
center_x = torch.arange(sub_sample / 2, feature_map_size * sub_sample, sub_sample)
center_y = torch.arange(sub_sample / 2, feature_map_size * sub_sample, sub_sample)

idx = 0
anchor_center = np.zeros((800, 800, 3), dtype=np.uint8)
center = torch.empty([feature_map_size * feature_map_size, 2])
for i in range(feature_map_size):
    for j in range(feature_map_size):
        cv2.circle(anchor_center, (int(center_x[i]), int(center_y[j])), 2, (0, 255, 255), -1)
        # center[idx, 1] = center_x[i]
        # center[idx, 0] = center_y[j]
        # idx += 1

plt.figure(figsize=(10, 10))
plt.imshow(anchor_center)
plt.show()

# 25x25x9 array of 4 coordinates of all anchor boxes
x_vec, y_vec = torch.meshgrid(center_x, center_y)
x_vec = x_vec.flatten()
y_vec = y_vec.flatten()

h_vec = sub_sample * torch.mm(torch.sqrt(anchor_ratio).view(3, 1), anchor_scale.view(1, 3)).flatten()
w_vec = sub_sample * torch.mm(torch.sqrt(1./anchor_ratio).view(3, 1), anchor_scale.view(1, 3)).flatten()



anchors = torch.zeros((feature_map_size * feature_map_size * len(anchor_ratio) * len(anchor_scale), 4))
index = 0
for c in center:
    center_y, center_x = c
    for i in range(len(anchor_ratio)):
        for j in range(len(anchor_scale)):
            h = sub_sample * anchor_scale[j] * torch.sqrt(anchor_ratio[i])
            w = sub_sample * anchor_scale[j] * torch.sqrt(1. / anchor_ratio[i])

            anchors[index, 0] = center_y - h / 2.
            anchors[index, 1] = center_x - w / 2.
            anchors[index, 2] = center_y + h / 2.
            anchors[index, 3] = center_x + w / 2.
            index += 1

print(anchors.shape)  # (22500, 4)

# Example of anchor boxes at (400, 400)
an_400 = anchor_center.clone()
for i in range(11470, 11478):
    cv2.rectangle(an_400.numpy(), (int(anchors[i][1]), int(anchors[i][0])), (int(anchors[i][3]), int(anchors[i][2])),
                  (255, 255, 255), 2)
plt.figure()
plt.imshow(an_400)
plt.show()

# Visualization of all the anchor boxes covering the input image
an_all = anchor_center.clone()
for i in range(len(anchors)):
    if int(anchors[i][0]) > 0 and int(anchors[i][1]) > 0 and int(anchors[i][2]) > 0 and int(anchors[i][3]) > 0:
        if int(anchors[i][0]) < 800 and int(anchors[i][1]) < 800 and int(anchors[i][2]) < 800 and int(
                anchors[i][3]) < 800:
            cv2.rectangle(an_all.numpy(), (int(anchors[i][1]), int(anchors[i][0])), (int(anchors[i][3]), int(anchors[i][2])),
                          (255, 255, 255), 1)
plt.figure(figsize=(15, 12))
plt.imshow(an_all)
plt.show()

# ---------------------- ANCHOR BOX LABELLING ----------------------
#  Now that we have defined our anchors, we need to create anchor targets out of them
# Let us first assign the labels and location of the objects w.r.t the anchor to each and every anchor

# We assign a positive label to
# a) The anchor with the highest IoU overlap with ground truth box
# b) Anchors that has IoU higher than 0.7 with the ground truth box
# Thus it's important to note that a single ground truth can assign positive labels to multiple anchors

# We assign a negative label to
# a) Anchors with IoU less than 0.3 for all ground truth boxes
# Anchors that are not positive or negative do not contribute to the training objective

bounding_box = torch.tensor([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=torch.float32)
labels = torch.tensor([6, 8], dtype=torch.int8)  # 0 represents background

# Let's first find all valid anchor boxes (recall boxes that go beyond the image boundaries are not valid)
index_inside = torch.where(((anchors[:, 0] >= 0) &
                           (anchors[:, 1] >= 0) &
                           (anchors[:, 2] <= 800) &
                           (anchors[:, 3] <= 800)), anchors, anchors)
index_inside = np.where((anchors[:, 0] >= 0) &
                           (anchors[:, 1] >= 0) &
                           (anchors[:, 2] <= 800) &
                           (anchors[:, 3] <= 800))[0]
print(index_inside.shape)  # (8940,)

# Creating an empty label array
label = np.empty((len(index_inside),), dtype=np.int32)
label.fill(-1)  # we fill them with -1 to begin with
print(label.shape)  # (2257,)

# An array of valid anchor boxes using the index we found above
val_anchors = anchors[index_inside]
print(val_anchors.shape)  # (2257, 4)

# Now we need to calculate the IoU with the ground truth to label the anchor boxes
ious = np.empty((len(val_anchors), len(bounding_box)), dtype=np.float32)  # iou per ground truth bbox
ious.fill(0)
print(bounding_box)

for num1, i in enumerate(val_anchors):
    ya1, xa1, ya2, xa2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)

    for num2, j in enumerate(bounding_box):
        yb1, xb1, yb2, xb2 = j
        box_area = (yb2 - yb1) * (xb2 - xb1)

        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)

            iou = inter_area / (anchor_area + box_area - inter_area)  # Intersection/Union

        else:  # no overlap
            iou = 0

        ious[num1, num2] = iou

print(ious.shape)

# Now that we've calculated the ious for each anchor box and ground truth box,
# it's time to label based on the rules defined above

# Finding highest IoU for each ground truth box (per column)
gt_argmax_ious = ious.argmax(axis=0)  # Search for max iou value for ground truth 1, and ground truth 2
print(gt_argmax_ious)

# Highest IoU value per ground truth (i.e box1's highest iou, box2's highest iou etc)
gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
print(gt_max_ious)

# Finding highest IoU for each anchor box (per row)
argmax_ious = ious.argmax(axis=1)
print(argmax_ious.shape)
print(argmax_ious)

# The max IoU value for each anchor box
max_ious = ious[np.arange(len(index_inside)), argmax_ious]
print(max_ious)

# Find where the anchor(s) which gave the highest IoU value with each ground truth is
# could be more than len(bbox) if more than one anchor has the same max_iou value for that truth
gt_argmax_ious = np.where(ious == gt_max_ious)[0]
print(gt_argmax_ious)

# Now finding the locations of the anchor box that satisfies the conditions
pos_iou_threshold = 0.7
neg_iou_threshold = 0.3

# Anchor which has IoU of less than 0.3 with ALL the ground truths are considered negative
label[max_ious < neg_iou_threshold] = 0

# Anchor which has the highest IoU with the ground truth is labeled positive
label[gt_argmax_ious] = 1

# Anchor which has IoU greater than 0.7 with a ground truth is labeled positive
label[max_ious >= pos_iou_threshold] = 1

# ---------------------- CREATING LOCATION TARGETS FOR RPN ----------------------
# Each mini-batch arises from a single image that contains many positive and negative example anchors,
# but this will biased towards the negative samples as they dominate.
# Instead, we randomly sample 256 anchors in an image to compute the loss function of a mini-batch,
# where the sampled positive and negative anchors have a ratio of up to 1:1.
# If there are fewer than 128 positive samples in an image, we pad the mini-batch with negative ones.

pos_ratio = 0.5
n_sample = 256

# POSITIVE SAMPLES
# We only want 128 positives, so if there are more than that, randomly disable some of them
n_pos = n_sample * pos_ratio
pos_index = np.where(label == 1)[0]
if len(pos_index) > n_pos:
    disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
    label[disable_index] = -1

# NEGATIVE SAMPLES
n_neg = n_sample - len(pos_index)
neg_index = np.where(label == 0)[0]
if len(neg_index) > n_neg:
    disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
    label[disable_index] = -1

# Assigning locations to anchor boxes
# We're finding out the location of the ground truth boxes w.r.t to the associated anchor box
max_iou_bbox = bounding_box[argmax_ious]  # Finding which bbox was max IoU for each anchor boxes
print(max_iou_bbox)

# We change our origin for comaparison (anchor box)
height = val_anchors[:, 2] - val_anchors[:, 0]
width = val_anchors[:, 3] - val_anchors[:, 1]
center_y = val_anchors[:, 0] + 0.5 * height
center_x = val_anchors[:, 1] + 0.5 * width

# We change the origin for comparison (ground truth box)
base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width

# Just to make sure height and width are not zero
eps = np.finfo(height.dtype).eps  # epsilon, very small number not zero
height = np.maximum(height, eps)
width = np.maximum(width, eps)

# Parameterization of FASTER-RCNN
# Offset in center position, and offset in width and height (anchor <-> ground truth)
dy = (base_ctr_y - center_y) / height
dx = (base_ctr_x - center_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)

anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
print(anchor_locs)

# Final labels for the anchors
anchor_label = np.empty((len(anchors),), dtype=label.dtype)
anchor_label.fill(-1)
anchor_label[index_inside] = label
print(anchor_label.shape)   # (5625,)

# Final location metric of anchors
anchor_locations = np.empty((len(anchors), anchors.shape[1]), dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[index_inside, :] = anchor_locs
print(anchor_locations.shape)   # (5625, 4)

# 'anchor_label' and 'anchor_locations' will be used as targets to the Region Proposal Network

# So 'anchor_label' tells you
# 1. Which anchors are going to be used for training,           ( label != -1 )
# 2. Which anchors are positive (high IoU with ground truth)    ( label == 1 )
# 3. Which anchors are negative (basically background)          ( label == 0 )

# And 'anchor_locations' tell you
# The x_center, y_center, width, height offset of the anchor box to the ground truth
# TODO : I'm still quite unsure about 'anchor locations' look it up


# ---------------------- REGION PROPOSAL NETWORK ----------------------
# Features coming out of the RPN are fed into two sibling layers:
# A box regression layer
# A box classification layer

mid_channels = 512
in_channels = 512
n_anchor = 9

conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
reg_layer = nn.Conv2d(mid_channels, n_anchor*4, 1, 1, 0)
cls_layer = nn.Conv2d(mid_channels, n_anchor*2, 1, 1, 0)

# Initialize the layers with weights
# conv sliding layer
conv1.weight.data.normal_(0, 0.01)
conv1.bias.data.zero_()

# Regression layer
reg_layer.weight.data.normal_(0, 0.01)
reg_layer.bias.data.zero_()

# classification layer
cls_layer.weight.data.normal_(0, 0.01)
cls_layer.bias.data.zero_()

# Recall that the out_map is the resulting feature map from the vgg feature extractor
x = conv1(out_map)
pred_anchor_locs = reg_layer(x)
pred_cls_scores = cls_layer(x)

print(pred_cls_scores.shape, pred_anchor_locs.shape)  # torch.Size([1, 18, 25, 25]) torch.Size([1, 36, 25, 25])

# We want to reshape the tensor tobe able to compare with our anchor_locations array
# which has shape [1, 5625, 4]
# So first we permute to change the dimension order
# and we would like to reshape it, but one dimension (the 0th) spans across two contiguous subspaces
# and hence, we need to apply .contiguous() first before we apply .view (which is pytorch reshape)
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
print(pred_anchor_locs.shape)  # torch.Size([1, 5625, 4])

pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
print(pred_cls_scores.shape)  # torch.Size([1, 5625, 18])

objectness_score = pred_cls_scores.view(1, 25, 25, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
print(objectness_score.shape)  # torch.Size([1, 5625])

pred_cls_scores = pred_cls_scores.view(1, -1, 2)
print(pred_cls_scores.shape)  # torch.Size([1, 5625, 2])

# So we've got the output from the RPN network where
# pred_cls_score and pred_anchor_locs are the outputs which are going to be used to find the loss and update the weights

# pred_cls_score and objectness_score are going to be used as inputs to the 'proposal layer'
# which generates a set of proposals used by the ROI network later on


# ---------------------- GENERATING PROPOSALS ----------------------
# RPN proposals of regions take the following parameters
# Training vs Testing
# Non Maximal Suppression (nms) threshold
# n_train_pre_nms (number of bounding boxes)
# n_train_post_nms
# n_test_pre_nms
# n_test_post_nms
# minimum height of object required to create a proposal

nms_thresh = 0.7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_size = 16

# First we convert the loc predictions from the rpn network to bbox format [y1, x1, y2, x2]
# Converting anchors from bbox to ctr_x, ctr_y, w, h
anc_height = anchors[:, 2] - anchors[:, 0]
anc_width = anchors[:, 3] - anchors[:, 1]
anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

# Make the tensors into numpy arrays
pred_anchor_locs_numpy = pred_anchor_locs[0].data.numpy()
objectness_score_numpy = objectness_score[0].data.numpy()

dy = pred_anchor_locs_numpy[:, 0::4]
dx = pred_anchor_locs_numpy[:, 1::4]
dh = pred_anchor_locs_numpy[:, 2::4]
dw = pred_anchor_locs_numpy[:, 3::4]

# reverse the parameterization
center_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
center_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
h = np.exp(dh) * anc_height[:, np.newaxis]
w = np.exp(dw) * anc_width[:, np.newaxis]

# Convert to bbox format
roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=pred_anchor_locs_numpy.dtype)
roi[:, 0::4] = center_y - 0.5 * h
roi[:, 1::4] = center_x - 0.5 * w
roi[:, 2::4] = center_y + 0.5 * h
roi[:, 3::4] = center_x + 0.5 * w

# Clipping the boxes to the image so that it doesn't go beyond the image borders
img_size = (800, 800)
roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

# Removing too small boxes
hs = roi[:, 2] - roi[:, 0]
ws = roi[:, 3] - roi[:, 1]
keep = np.where((hs >= min_size) & (ws >= min_size))[0]
roi = roi[keep, :]

# get score and sort
score = objectness_score_numpy[keep]
order = score.ravel().argsort()[::-1]
print(order)

# Take top pre_nms_topN
order = order[:n_train_pre_nms]
roi = roi[order, :]

print(roi.shape)
print(roi)

# Applying Non Maximal Suppression
# 1. Finding areas of all the boxes
y1 = roi[:, 0]
x1 = roi[:, 1]
y2 = roi[:, 2]
x2 = roi[:, 3]

area = (x2 - x1 + 1) * (y2 - y1 + 1)

# 2. Get order of probability score
order = score.argsort()[::-1]
#order = order[inds + 1]

keep = []

while order.size > 0:
    i = order[0]
    keep.append(i)

    # Find intersecting area
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h

    # if intersecting area is beyond some nms threshold, do not add to keep
    overlap = inter / (area[i] + area[order[1:]] - inter)
    inds = np.where(overlap <= nms_thresh)[0]
    order = order[inds + 1]     # Delete entry from order if overlap is over nms threshold

print(len(keep))
keep = keep[:n_train_post_nms]  # while training/testing , use accordingly
roi = roi[keep]  # the final region proposals

# This final region proposal is used as the input to Faster-RCNN to try
# 1.  Predict the object location w.r.t to the proposed box
# 2.  Classify the object in the proposal


# ----------------------  PROPOSAL TARGETS ----------------------

#  The Faster RCNN network takes the region proposals, ground truth boxes, and their labels as inputs
# The following parameters are passed to:
# - Number of samples to sample from roi
# - number of positive examples out of n_samples
# - the minimum overlap of region proposal with any ground truth to be considered positive label
# - The overlap value bound to consider a region proposal as negative (background)

n_sample = 128
pos_ratio = 0.25
pos_iou_thresh = 0.5
neg_iou_thresh_hi = 0.5
neg_iou_thresh_lo = 0.0

# We first find the IoU of each ground truth object with the region proposals
ious = np.empty((len(roi), 2), dtype=np.float32)
ious.fill(0)
for num1, i in enumerate(roi):
    ya1, xa1, ya2, xa2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bounding_box):
        yb1, xb1, yb2, xb2 = j
        box_area = (yb2- yb1) * (xb2 - xb1)

        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])

        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
            iou = iter_area / (anchor_area + box_area - iter_area)
        else:
            iou = 0.

        ious[num1, num2] = iou
print(ious.shape)

# Which ground truth have a high IoU for each region proposal?
# Find Max IoU too
gt_assignment = ious.argmax(axis=1)
max_iou = ious.max(axis=1)
print(gt_assignment)
print(max_iou)

# Assign the labels to proposals
gt_roi_label = labels[gt_assignment]
print(gt_roi_label)

# Select foreground ROI (positive)
pos_index = np.where(max_iou >= pos_iou_thresh)[0]
pos_roi_per_this_image = n_sample * pos_ratio
pos_roi_per_this_image = int(min(pos_roi_per_this_image, pos_index.size))
if pos_index.size > 0:
    pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)
print(pos_roi_per_this_image)
print(pos_index)

# Select background ROI
neg_index = np.where((max_iou < neg_iou_thresh_hi) & (max_iou >= neg_iou_thresh_lo))[0]
neg_roi_per_this_image = n_sample - pos_roi_per_this_image
neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
if neg_index.size > 0:
    neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)
print(neg_roi_per_this_image)
print(neg_index)

# We collect both positive and negative sample index, their respective labels, and region proposals
keep_index = np.append(pos_index, neg_index)
gt_roi_labels = gt_roi_label[keep_index]
gt_roi_labels[pos_roi_per_this_image:] = 0  # negative labels --> 0
sample_roi = roi[keep_index]
print(sample_roi.shape)

# Pick the ground truth objects for these sample rois
bbox_for_sampled_roi = bounding_box[gt_assignment[keep_index]]
print(bbox_for_sampled_roi.shape)

height = sample_roi[:, 2] - sample_roi[:, 0]
width = sample_roi[:, 3] - sample_roi[:, 1]
center_y = sample_roi[:, 0] + 0.5 * height
center_x = sample_roi[:, 1] + 0.5 * width

base_height = bbox_for_sampled_roi[:, 2] - bbox_for_sampled_roi[:, 0]
base_width = bbox_for_sampled_roi[:, 3] - bbox_for_sampled_roi[:, 1]
base_ctr_y = bbox_for_sampled_roi[:, 0] + 0.5 * base_height
base_ctr_x = bbox_for_sampled_roi[:, 1] + 0.5 * base_width

eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)

dy = (base_ctr_y - center_y) / height
dx = (base_ctr_x - center_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)

gt_roi_locs = np.vstack((dy, dx, dh, dw)).transpose()
print(gt_roi_locs)

# So now we've got the ground truth roi locations and the ground truth roi labels
# We now need to design the FASTER RCNN network to predict the locs and labels.


# ---------------------- FASTER-RCNN ----------------------
# We need to do ROI pooling to match the different size rois into one uniform size that a network can handle
# We're basically performing max pooling on inputs of non-uniform sizes to obtain fixed size feature maps

# The ROI pooling layer takes in two inputs
# 1. A fixed size feature map obtained from a DCNN with several convolutions and max pooling layers
# 2. A Nx5 matrix representing a list of regions of interest, N being the number of ROIs
# the 5 stands for the image index + 4 coordinates

# For every region of interest from the input list, it takes a section of the input map and scales it

# we use the sample_roi as input to the roi_pooling layer which has [N, 4] dimensions
rois = torch.from_numpy(sample_roi).float()
roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
roi_indices = torch.from_numpy(roi_indices).float()
print(rois.shape, roi_indices.shape)

# Add dimension
indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
indices_and_rois = xy_indices_and_rois.contiguous()
print(xy_indices_and_rois.shape)

# Now that we've matched the dimensions, we can pass it through the roi_pooling layer
size = (7, 7)
adaptive_max_pool = nn.AdaptiveAvgPool2d((size[0], size[1]))

output = []
rois = indices_and_rois.data.float()
rois[:, 1:].mul_(1/sub_sample)
rois = rois.long()
num_rois = rois.size(0)

for i in range(num_rois):
    roi = rois[i]
    im_idx = roi[0]
    im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
    output.append(adaptive_max_pool(im))

output = torch.cat(output, 0)
print(output.size())  # torch.Size([128, 512, 7, 7])

k = output.view(output.size(0), -1)
print(k.shape)   # torch.Size([128, 25088])

roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096),
                                     nn.Linear(4096, 4096)])

# 2 output layers, the 'locations' and the 'score'
cls_loc = nn.Linear(4096, 21*4)  # VOC 20 classes + 1 background, each with 4 coordinates

# Initialize weights and bias
cls_loc.weight.data.normal_(0, 0.01)
cls_loc.bias.data.zero_()

score = nn.Linear(4096, 21)

# Passing the output of the roi_pooling layer into the network
k = roi_head_classifier(k)
roi_cls_loc = cls_loc(k)
roi_cls_score = score(k)
print(roi_cls_loc.shape, roi_cls_score.shape)  # torch.Size([128, 84]) torch.Size([128, 21])

# ---------------------- LOSS FUNCTIONS ----------------------
# Now that we've completed the setup, we can calculate loss functions for the two outputs
# We have roi_cls_loc the regression head
# and the roi_cls_score the classification head

print(pred_anchor_locs.shape)
print(pred_cls_scores.shape)
print(anchor_locations.shape)
print(anchor_label.shape)

rpn_loc = pred_anchor_locs[0]
rpn_score = pred_cls_scores[0]

gt_rpn_loc = torch.from_numpy(anchor_locations)
gt_rpn_score = torch.from_numpy(anchor_label)

print(rpn_loc.shape, rpn_score.shape, gt_rpn_loc.shape, gt_rpn_score.shape)

# pred_cls_scores and anchor_labels are the predicted objectness score and actual objectness of the RPN network

# For classification we use cross-entropy loss
# For regression we use smooth L1 (as values of predicted region not bounded)

# Classification loss
rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_score.long(), ignore_index=-1)
print(rpn_cls_loss)  # tensor(0.6935, grad_fn=<NllLossBackward>)

# Regression loss applied to bounding boxes with positive label
pos = gt_rpn_score > 0
mask = pos.unsqueeze(1).expand_as(rpn_loc)
print(mask.shape)  # torch.Size([5625, 4])

mask_loc_preds = rpn_loc[mask].view(-1, 4)
mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)
print(mask_loc_preds.shape, mask_loc_preds.shape)  # torch.Size([3, 4]) torch.Size([3, 4])

x = torch.abs(mask_loc_targets.float() - mask_loc_preds)
rpn_loc_loss = ((x < 1).float() * 0.5 * x**2) + ((x >= 1).float() * (x-0.5))
print(rpn_loc_loss.sum())  # tensor(0.1891, grad_fn=<SumBackward0>)

# We calculate the final loss by combining both rpn_loc_loss and rpn_cls_loss
# yet the classification loss is for all bounding boxes, and the regression loss is only on positive boxes
# So there is a separate hyperparamter lambda to weigh it

rpn_lambda = 10.
N_reg = (gt_rpn_score > 0).float().sum()
rpn_loc_loss = rpn_loc_loss.sum() / N_reg  # Normalize loss over number of all boxes
rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)  # Adding weight to regression loss
print(rpn_loss)  # tensor(1.3238, grad_fn=<AddBackward0>)
# And hence we got the RPN Loss sorted


# Now to look at the RCNN loss
# we've got the following predicted
print(roi_cls_loc.shape)  # torch.Size([128, 84])
print(roi_cls_score.shape)  # torch.Size([128, 21])

# and the following ground truth
print(gt_roi_locs.shape)  # (128, 4)
print(gt_roi_labels.shape)  # (128,)
gt_roi_loc = torch.from_numpy(gt_roi_locs)
gt_roi_label = torch.from_numpy(np.float32(gt_roi_labels)).long()
print(gt_roi_loc.shape, gt_roi_label.shape) # torch.Size([128, 4]) torch.Size([128])

# Classification loss
roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_label, ignore_index=-1)
print(roi_cls_loss.shape)

# Regression loss
n_sample = roi_cls_loc.shape[0]
roi_loc = roi_cls_loc.view(n_sample, -1, 4)
print(roi_loc.shape)  # torch.Size([128, 21, 4])

roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label]
print(roi_loc.shape)  # torch.Size([128, 4])

y = torch.abs(gt_roi_loc - roi_loc)
roi_loc_loss = ((y < 1).float() * 0.5 * y**2) + ((y >= 1).float() * (y-0.5))
roi_loc_loss = roi_loc_loss.sum()
print(roi_loc_loss)

roi_lambda = 10.
roi_loss = roi_cls_loss + (roi_lambda * roi_loc_loss)
print(roi_loss)

# For one iteration, the total loss is teh sum of the RPN loss and the Fast-RCNN loss
total_loss = rpn_loss + roi_loss
print(total_loss)
