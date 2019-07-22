"""
With the advent of deep learning and neural network models, a lot of it's applications have been founded in computer vision. A statistical model based on the way our brain is wired is used to emulate many visual tasks that humans find so natural almost to a trivial sense.

The first task was to deal with how we recognise 'things' just by looking at them. We see an object and we instantly classify it's class(what it is). Hereby an image Classifer was developed using convolutional neural networks. These type of neural networks utilised image convolutions to embed spatial data in it's attempt to extract features from the image, which can in turn be used to distinguish between objects of different classes.
Now classifying an image is all good, but what if there are more than one object in an image? Or what about meaning of locality or dispersity in our visual comprehension? How do we know where the object is and how much space it takes up? These are questions that naturally arise once you have a classifier, and the logical step is to develop an Object Detector. I guess the ultimate end goal is to classify each pixel into it's class, and acheive pixel-wise localization of an object, but first a much simpler task needs to be dealt with. Finding out the bounding boxes that captures an object, and classifying what object it's holding in.

Bounding boxes present us with a very simple yet intuitive way to localize objects in an image, and differentiate between two different objects. To acheive instance segmentation, you would need to differentiate between each object and classify the pixels within as well. So this gives us a good starting point.

One can easily notice that developing an object detector then, is just adding one more component to the classifiers devloped above. Something that can localize the objects as bounding boxes, a Region Proposing Algorithm. Once you've identified some target regions that you think are objects, the classifer can tell us what the object is. This is what happend with the development of two-stage object detectors (There are one-stage object detectors which uses a grid based approach to predict objects too).

Faster RCNN is the first Object Detector in the line of R-CNNs (two stage detectors) that uses only neural networks to detect an object. It's predecessors, RCNN and Fast RCNN still uses the traditional 'Selective Search' algorithm based on graph based segmentations. This traditional method is slow compared to running through a neural network, and thus Faster RCNN replaces the region proposal algorithm with a Region Proposal Network.

"""

"""
Feature Extractor
As seen in the diagram above, we first start off by getting a feature map of our input image. We can do this in many ways, but one of the easiest way to do so would be to utilize a pre-existing classifier model. Modern deep learning classifers are made up of two distinct parts, 1) The Feature Extractor and 2) The Classifer So we could simply just use the first part of an image classifer and get a feature map.

In this example we will use the VGG16 backbone

"""
# Importing all the necessary modules

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import cv2
import matplotlib.pyplot as plt
import random
import time

time1 = time.time()

# Choose Device for Torch
# device='cpu'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Let us assume our neural network has the following properties
# 1) It takes in a 800 x 800 three channel image
# 2) Bounding boxes are given in the format [y1, x1, y2, x2]
# 3) Labels are given by integers, -1 : Not used for training, 0: Background, 1: Object etc..


# Define input size
input_size = (800, 800)

# A typical image would look like this
# Torch Tensors have the order [N, C, H, W]
image = torch.zeros((1, 3, *input_size)).float()

# Bounding boxes and it's labels
bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]])  # [y1, x1, y2, x2]
labels = torch.LongTensor([6, 8])

# This will define our feature map size. The smaller the sub_sample is the bigger the feature map is -> more computation
sub_sample = 16

# Number of classes
num_class = 2

# We will use the two dummy image and bounding boxes to illustrate the full loss function calculation

# Let's get the VGG16 model

model = torchvision.models.vgg16(pretrained=True)

# Getting the feature extractor part of the model
feature_extractor = list(model.features)
print("Layers in the Feature Extractor part of VGG16 : \n\n", feature_extractor)

# Now we have to trim the feature extractor to fit our sub_sample ratio
req_features = []
k = image.clone()

# Iterate over the layers in the feature extractor
for layer in feature_extractor:
    k = layer(k)  # Pass the image through the layers

    # Until the output feature map size satisfies our sub_sample ratio
    if k.size()[2] < input_size[0] // sub_sample:
        break

    # Save the required layers
    req_features.append(layer)
    out_channels = k.size()[1]

print("\nNumber of layers we will be using to get the Feature Map : ", len(req_features))
print("\nNumber of channels(filters) in the resulting forward pass : ", out_channels)

# Now we can define our feature extractor with the layers we've got from above
faster_rcnn_feature_extractor = nn.Sequential(*req_features)
faster_rcnn_feature_extractor = faster_rcnn_feature_extractor.to(device)

image = image.to(device)

# Extracting the features we get
out_map = faster_rcnn_feature_extractor(image)
print(out_map.size())

"""
Since we have pooled our image from 800 to 50, we have a sub_sampling ratio of 16 (16*50 = 800).

This means that every pixel in the output feature map we just got above corresponds to 16x16 pixels in the input image

As this was an operation under convolution, the spatial structure is maintained. That is, the first pixel of the 50x50 feature map relates to the first 16x16 window

"""

"""
Region Proposal Network

Anchor Box Generation

We start creating the Region Proposal Network by defining Anchor Boxes. They are the building blocks of the region proposals, as we define the proposals as a transformation of these anchor boxes.

The method in which you define the anchor boxes can affect the performance and speed of your network as well. But here we will just use the general case of anchor boxes with 3 scales and 3 h-w ratios.

We generate these anchor box for every 16x16 window of the image (i.e one for every pixel in the feature map). Hence, a total of 3 scales * 3 ratios * 50(height) * 50(width) = 22500 anchor boxes will be made. At least this narrows down from the infinite possibilties of choosing regions from an image.

For each pixel position of the feature map, an anchor will have the shape (9, 4) --> 9 from the combination of scales and ratios, and 4 coordinates
"""

# Setting Anchor ratios and Anchor scales
anchor_ratio = torch.Tensor([0.5, 1, 2]).to(device)
anchor_scale = torch.Tensor([8, 16, 32]).to(device)

# We will generate anchor boxes for every center of the 16 x 16 pixel window, so first we get the centers of each window

feature_map_size = input_size[0] // sub_sample

center_y = torch.arange(sub_sample / 2, feature_map_size * sub_sample, sub_sample).to(device)
center_x = torch.arange(sub_sample / 2, feature_map_size * sub_sample, sub_sample).to(device)

# Visualizing the anchor centers and also collecting the center coordinates
anchor_cen_vis = np.zeros((*input_size, 3), dtype=np.uint8)
idx = 0

center = torch.empty([feature_map_size * feature_map_size, 2])
for i in range(feature_map_size):
    for j in range(feature_map_size):
        cv2.circle(anchor_cen_vis, (int(center_x[i]), int(center_y[j])), 2, (0, 255, 255), -1)
#         center[idx, 1] = center_x[i]
#         center[idx, 0] = center_y[j]
#         idx += 1

plt.figure(figsize=(10, 10))
plt.imshow(anchor_cen_vis)
plt.show()

# Calculate all the anchor boxes with the given anchor scales and ratios
x_vec, y_vec = torch.meshgrid(center_x, center_y)
x_vec = x_vec.flatten()
y_vec = y_vec.flatten()

h_vec = sub_sample * torch.mm(torch.sqrt(anchor_ratio).view(3, 1), anchor_scale.view(1, 3)).flatten()
w_vec = sub_sample * torch.mm(torch.sqrt(1. / anchor_ratio).view(3, 1), anchor_scale.view(1, 3)).flatten()

anc_0 = (y_vec.view(y_vec.size(0), 1) - h_vec / 2.).flatten()
anc_1 = (x_vec.view(x_vec.size(0), 1) - w_vec / 2.).flatten()
anc_2 = (y_vec.view(y_vec.size(0), 1) + h_vec / 2.).flatten()
anc_3 = (x_vec.view(x_vec.size(0), 1) + w_vec / 2.).flatten()

anchors = torch.stack([anc_0, anc_1, anc_2, anc_3], dim=1)
print(anchors.shape)

# We initialize an empty array of anchors to fill it in with calculated anchors
# anchors = torch.zeros((feature_map_size * feature_map_size * len(anchor_ratio) * len(anchor_scale), 4)).to(device)

# index = 0

# # Iterate over all the centers displayed above
# for c in center:
#     ctr_y, ctr_x = c

#     # For every anchor ratio
#     for i in range(len(anchor_ratio)):

#         # and For every anchor scale
#         for j in range(len(anchor_scale)):

#             # Create an anchor box and add to the 'anchors' array
#             h = sub_sample * anchor_scale[j] * torch.sqrt(anchor_ratio[i])
#             w = sub_sample * anchor_scale[j] * torch.sqrt(1. / anchor_ratio[i])

#             anchors[index, 0] = ctr_y - h / 2.
#             anchors[index, 1] = ctr_x - w / 2.
#             anchors[index, 2] = ctr_y + h / 2.
#             anchors[index, 3] = ctr_x + w / 2.
#             index += 1

# print('Anchor array shape : ', anchors.shape)
# print(anchors[0:10])
# fortime = time.time()

# print(fortime - vectime2)
# print(vectime2 - vectime)
# print(torch.all(torch.eq(anchors, anchors2)))

# Visualizing created anchors

center_anchors = anchor_cen_vis.copy()

for anc_idx in range(1275 * 9, 1275 * 9 + 9):
    y1 = int(anchors[anc_idx][0])
    x1 = int(anchors[anc_idx][1])
    y2 = int(anchors[anc_idx][2])
    x2 = int(anchors[anc_idx][3])

    cv2.rectangle(center_anchors, (x1, y1), (x2, y2), (255, 255, 255), 2)

plt.figure(figsize=(10, 10))
plt.imshow(center_anchors)
plt.show()

# Visualizing all anchors covering the input image

all_anchors = anchor_cen_vis.copy()
for i in range(len(anchors)):
    if 800 > int(anchors[i][0]) > 0 and 800 > int(anchors[i][1]) > 0 and 800 > int(anchors[i][2]) > 0 and 800 > int(
            anchors[i][3]) > 0:
        cv2.rectangle(all_anchors, (int(anchors[i][1]), int(anchors[i][0])), (int(anchors[i][3]), int(anchors[i][2])),
                      (255, 255, 255), 1)

cv2.imwrite("img/allanchors.png", all_anchors)

"""
Anchor Box Labelling

Now that we have defined our anchors, we need to create anchor targets out of them 
We will first assign the labels(Object or not) --> Classification and location of the objects --> Regression(where in the image) w.r.t the anchor to each and every anchor

It is imperative to understand that the labels here are not the final classification labels for the object. 
These are positive/negative labels for region proposals, labels tagging whether a region has an object or not *whatever it is***.

The following is the Classification labelling criteria:

We assign a positive label to

a) The anchor with the highest IoU overlap with the ground truth box 
b) Anchors that has an IoU greater than 0.7 with the ground truth box


We assign a negative label to

c) Anchors with IoU less than 0.3 for all ground truth boxes

Anchors that are not positive or negative do not contribute to the training objective



Anchor Box 'Objectiveness' Classification labelling
"""
# Recall the dummy bounding boxes and labels we've assigned in the beginning

# bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]])  # [y1, x1, y2, x2]
# labels = torch.LongTensor([6, 8])

# We are going to try and label our anchors for them
bbox = bbox.to(device)
labels = labels.to(device)

# First we will filter our anchors so that we only have 'valid' anchors.
# By valid, we want anchor boxes that do not go beyond the image boundaries

idx_in = ((anchors[:,0] >= 0) & (anchors[:,1] >= 0) & (anchors[:,2] <= 800) &(anchors[:,3] <= 800)).nonzero().transpose(0, 1)[0]

print('Shape of valid indicies : ', idx_in.shape)


# This will hold all 'valid' anchors
val_anchors = anchors[idx_in]

print('Shape of valid anchors Tensor : ', val_anchors.shape)

# Creating an empty label array
label = torch.empty((len(idx_in),)).int()
label.fill_(-1)
print(label.shape)

### Now that we have valid anchors, an an empty label array to fill in, we need to calculate the IoUs to find out what label to assign to each valid anchor!
# IoU, or Intersection over Union is simply defined as the ratio between the intersection of two bouding boxes and the union of the two.


# Recall that we have 2 bounding boxes in our ground truth
print('Bounding Boxes : ', bbox)

# Calculating the Intersection over Union between the ground truth boxes and the anchor boxes
# First we get each coordinates as vectors separately
ya1, xa1, ya2, xa2 = torch.chunk(val_anchors, 4, dim=1)  # Valid Anchor Boxes
yb1, xb1, yb2, xb2 = torch.chunk(bbox, 4, dim=1)  # Ground truth Boxes

# Then we check the intersection by testing maximum/minimum
inter_x1 = torch.max(xa1, torch.transpose(xb1, 0, 1))  # This will result in (a n_anchor_box, 2) Tensor, which has the maximum for each ground truth box vs anchor box
inter_y1 = torch.max(ya1, torch.transpose(yb1, 0, 1))
inter_x2 = torch.min(xa2, torch.transpose(xb2, 0, 1))
inter_y2 = torch.min(ya2, torch.transpose(yb2, 0, 1))

# Calculating the intersection
inter_area = torch.max((inter_y2 - inter_y1 + 1), torch.Tensor([0, 0]).to(device)) * torch.max(
    (inter_x2 - inter_x1 + 1), torch.Tensor([0, 0]).to(device))
# The multiplication above is not matrix multiplication (the dimensions don't match anyway), it's scalar-wise multiplication
# The output has shape of (8940, 2), each column representing the area of intersection of each anchor box with corresponding bounding box (1st column = 1st bounding box) and so on

# Calculating the Union
anchor_area = (xa2 - xa1 + 1) * (ya2 - ya1 + 1)  # (8940, 1)
bbox_area = (xb2 - xb1 + 1) * (yb2 - yb1 + 1)  # (2, 1)

union_area = anchor_area + torch.transpose(bbox_area, 0, 1) - inter_area

# Calculating the IoU
iou = inter_area / union_area
print("\nShape of the array holding the IoUs with each bounding box truth : ", iou.shape)

# Now that we've calculated all the IoUs for each anchor boxes, we can start filling in the label array
# Recall that we assgin a positive label to

# a) The anchor with the highest IoU overlap with the ground truth box <br>
# b) Anchors that has an IoU greater than 0.7 with the ground truth box

# and a negative label to
# c) Anchors with IoU less than 0.3 for all ground truth boxes

# Therefore,
# 1. We need to find the highest IoU value for each ground truth box (a)
# 2. Find the maximum IoU per anchor box
# (if max_IoU > 0.7 then *at least one* IoU with a ground truth is greater than 0.7 IoU) (b)
# (if max_IoU < 0.3 then for *all* ground truth, the anchor has less than 0.3 IoU) (c)



# Finding the highest IoU for each ground truth box
gt_argmax_ious = torch.argmax(iou, dim=0)
gt_max_ious = iou[gt_argmax_ious, torch.arange(iou.shape[1])]
print('Highet IoU per ground truth--> \nbbox 1 at {} with IoU : {}\n bbox2 at {} with IoU : {}'.format(gt_argmax_ious[0], gt_max_ious[0], gt_argmax_ious[1], gt_max_ious[1]))

gt_argmax_ious = (iou == gt_max_ious).nonzero().transpose(0,1)[0]
print('\nAll of the indices that share the max value for each ground truth box : \n', gt_argmax_ious)


# Finding maximum IoU per anchor box
argmax_ious = torch.argmax(iou, dim=1)
max_ious = iou[torch.arange(len(idx_in)), argmax_ious]
print('\nMax IoUs per anchor box', max_ious)


# Labelling
pos_iou_threshold = 0.7
neg_iou_threshold = 0.3

label = label.to(device)

# Anchors which have the highest IoU with the ground truth is labeled positive
label[gt_argmax_ious] = 1

# Anchors which have IoU greater than 0.7 with 'a' ground truth is labeled positive
label[max_ious >= pos_iou_threshold] = 1

# Anchors which has IoU of less than 0.3 with ALL the ground truths are considered negative
label[max_ious < neg_iou_threshold] = 0

print('\nNumber of Positive Labels : {} \nNumber of Negative Labels : {} \nNumber of Non-Training Labels : {}'.format(len(label[label==1]),len(label[label==0]), len(label[label==-1])))

"""
Creating Location Targets for the Anchor Boxes

Now that we have sorted the targets(labels) for the classification problem above, the next step is to create the location targets for the regression problem.

We are interested in the location of the object relative to the anchor boxes.

Each mini-batch arises from a single image that contains many positive and negative example anchors, but this will biased towards the negative samples as they dominate.
Instead, we randomly sample 256 anchors in an image to compute the loss function of a mini-batch, where the sampled positive and negative anchors have a ratio of up to 1:1.
If there are fewer than 128 positive samples in an image, we pad the mini-batch with negative ones.
"""
# Random sampling of anchors per image

pos_ratio = 0.5
n_sample = 256


def reservoir_sample(n_tensor, k):
    reservoir = torch.empty(k).to(device)
    for idx, item  in enumerate(n_tensor):
        if idx < k:
            reservoir[idx] = item
        else:
            m = torch.randint(0, idx, (1,)).long()
            if m < k:
                reservoir[m] = item
    return reservoir


# Positive samples
n_pos = n_sample * pos_ratio
pos_index = (label == 1).nonzero().transpose(0, 1)[0]

# If there are more than 128 positive samples, remove some
if pos_index.size(0) > n_pos:
    # torch.randint will return random indices between 0 and pos_index.size(0) with size (len(pos_index) - n_pos)
    disable_index = reservoir_sample(pos_index, pos_index.size(0) - n_pos).long()
    label[disable_index] = -1  # Remove the excess

# Negative samples
n_neg = n_sample - pos_index.size(0)
neg_index = (label == 0).nonzero().transpose(0, 1)[0]

if neg_index.size(0) > n_neg:
    disable_index = reservoir_sample(neg_index, neg_index.size(0) - n_neg).long()
    label[disable_index] = -1  # Remove the excess

print('After Sampling to reduce bias >>')
print('\nNumber of Positive Labels : {} \nNumber of Negative Labels : {} \nNumber of Non-Training Labels : {}'.format(
    len(label[label == 1]), len(label[label == 0]), len(label[label == -1])))

# We will now assign the locations to the anchor boxes (labelling woohoo finally!)
# We assign the locations of the ground truth object to each anchor box with has the highest IoU with it

max_iou_bbox = bbox[argmax_ious]
print('These are the locations that we will assign to each anchor boxes : \n', max_iou_bbox)
# We get the ground truth boxes for all anchor boxes irrespective of it's label and we'll filter it later on

# We assign a parameterized version of the coordinates to the anchors
# The parameterization is as follows:
# t_{x} = (x - x_{a})/w_{a}  --> Gives normalized x_diff relative to anchor width
# t_{y} = (y - y_{a})/h_{a}  --> Gives normalized y_diff relatie to anchor height
# t_{w} = log(w/ w_a)        --> If w|h of gt is smaller than anchor gives negative (the smaller it is the stronger the parameterization as log(x) -> -inf as x->0)
# t_{h} = log(h/ h_a)        --> whereas if w|h is bigger, it's positive, but at a very shallow rate of increase

# So this is purely my opinion, but I think they used the log parameterization for 3 reasons
# 1. It gives the proper direction information for a ratio between widths. '1' should be the point where the parameterization turns from negative to positive and log(x) does that
# 2. Widths and heights are always positive which means that it's bounded on one side (0, inf), and therefore we need a function with an asymptote at x = 0, log(x) does that
# 3. We want to control the growth of numbers, and log(x) grows slower than y = x on the positive side, also it penalizses 'smallness'


# To calculate the parameterization, we first need to convert the [y1, x1, y2, x2] in to [ctr_x, ctr_y, width, height]
# Anchors
height = val_anchors[:, 2] - val_anchors[:, 0]
width = val_anchors[:, 3] - val_anchors[:, 1]
ctr_y = val_anchors[:, 0] + 0.5 * height
ctr_x = val_anchors[:, 1] + 0.5 * width

# Ground truth boxes
base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width


# Parameterization
# Insure height and width are > 0
eps = 0.00001
height = torch.max(height, torch.full_like(height, eps))   # Ensure height and width are > 0
width = torch.max(width, torch.full_like(width, eps))

dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = torch.log(base_height / height)
dw = torch.log(base_width / width)

anchor_locs = torch.stack((dy, dx, dh, dw), 1)

print('These are the paramterized locations that we will assign to each anchor boxes : \n', anchor_locs)

# Organizing our labels
print('FINAL RPN NETWORK TARGETS(LABELS)')
print('---------------------------------------')


# Final labels for the anchors
anchor_label = torch.empty((anchors.size(0),)).int().to(device)
anchor_label.fill_(-1)
anchor_label[idx_in] = label
print('Anchor Objectness Label : ', anchor_label.shape)
print('Number of Positive Labels : {} \nNumber of Negative Labels : {} \nNumber of Non-Training Labels : {}'
      .format(len(anchor_label[anchor_label==1]),len(anchor_label[anchor_label==0]), len(anchor_label[anchor_label==-1])))


# Final location labels of anchors
anchor_locations = torch.empty_like(anchors)
anchor_locations.fill_(0)
anchor_locations[idx_in, :] = anchor_locs  # Recall 'idx_in' are the index of the anchors that are 'valid' (i.e inside the 800x800 window)
print('\nAnchor Location Regression Target : ', anchor_locations.shape)

"""
Region Proposal Network Architecture

We have successfully computed the targets that we want for the Region Proposal Network above.
Now we need to define our RPN in order to compute the loss with the targets we got!

RPN is quite simple in it's structure. It contains a convolution module, which then is fed into sibling layers of

one regression layer -> predicts to location of the box inside the anchor
one classification layer -> predicting the 'objectness' of the proposed box
"""
# Recall that the feature map has 512 channels!
mid_channels = 512
in_channels = 512
n_anchor = 9


# The regression and classifcation layer comes after a convolution layer
conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1).to(device)  # Convolution layer with 3x3 window, 1 stride, 1 padding

# Box location regression layer
reg_layer = nn.Conv2d(mid_channels, n_anchor*4, 1, 1, 0).to(device)  # Convolution layer with 1x1 window (dimension reducing?)
# Box classification layer
cls_layer = nn.Conv2d(mid_channels, n_anchor*2, 1, 1, 0).to(device)  # Convolution layer with 1x1 window (dimension reducing?),
                                                                     #n_anchor*2 cause we'll be using softmax (multi-class), we can do 1 and sigmoid(binary)

# Initializing the layers with zero mean and 0.01 sd

# conv sliding layer
conv1.weight.data.normal_(0, 0.01)
conv1.bias.data.zero_()

# Regression layer
reg_layer.weight.data.normal_(0, 0.01)
reg_layer.bias.data.zero_()

# classification layer
cls_layer.weight.data.normal_(0, 0.01)
cls_layer.bias.data.zero_()

# As we have finished defining the RPN, we can now get the predictions by forward passing the output feature map we got from the feature extractor part

# Convolution layer
x = conv1(out_map)  # Recall that out_map was the feature map we got from part 1

# Getting Predictions
pred_anchor_locs = reg_layer(x)
pred_cls_scores = cls_layer(x)

# This is the prediction we get for the dummy image (a black 800x800 image)
print('pred_cls_scores : {}, \npred_anchor_locs : {}'.format(pred_cls_scores.shape, pred_anchor_locs.shape))

# As you can see, the output size is incomprehensible, it does not match with our generated labels above.
# So we shall reshape it! We want to reshape it into [1, 22500, 4]

# So first we permute to change the dimension order
# and we would like to reshape it, but one dimension (the 0th) spans across two contiguous subspaces
# and hence, we need to apply .contiguous() first before we apply .view (which is pytorch reshape)

# Reshaping anchor location predictions
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)   # -1 calculates for you
print('Reshaped pred_anchor_locs : ', pred_anchor_locs.shape)


# Reshaping anchor classification predictions
pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
objectness_score = pred_cls_scores.view(1, out_map.size(2), out_map.size(3), 9, 2)[..., 1].contiguous().view(1, -1)
print('Objectness score         : ', objectness_score.shape)


pred_cls_scores = pred_cls_scores.view(1, -1, 2)
print('Reshaped pred_cls_scores : ', pred_cls_scores.shape)  # the two is [1st class : Object, 2nd Class : Not Object]

"""
We will be using the pred_cls_scores and pred_anchor_locs to calculate loss for the RPN 
We will be using the pred_cls_scores and objectness_scre are used as inputs to the proposal layer -> proposal used by ROI network
"""

"""
Proposals for Fast-RCNN

The generation of proposals which will be used by the ROI network is next.

RPN proposals of regions take the following parameters 

Training vs Testing
Non Maximal Suppression (nms) threshold
n_train_pre_nms (number of bounding boxes) -n_train_post_nms
n_test_pre_nms
n_test_post_nms
minimum height of object required to create a proposal
"""
# Defining parameters used by the Proposal layer

nms_thresh = 0.7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_test_pre_nms = 6000
n_test_post_nms = 300
min_size = 16

"""
By proposals, it just means the predicted regions from the box regression layer above, and we give this proposed roi to the roi pooling layer

The following steps are taken to generate a region of interest proposals

Convert the location predictions from the rpn network to bbox [y1,x1, y2, y3] back again
Clip the predicted boxes to the images so that it doesn't go beyond the image
Remove predicted boxes that are too small
Sort all (proposal, score) pairs by score descending order
Take top pre_nms_topN
Apply Non Maximal Suppression
Take top post_nms_topN -> Final proposals
"""
# 1. Converting the location predictions back to bbox format

# We have to 'un'-parameterize it, and thus the inverse function would be:
# x = (w_{a} * ctr_x_{p}) + ctr_x_{a}
# y = (h_{a} * ctr_x_{p}) + ctr_x_{a}
# h = np.exp(h_{p}) * h_{a}
# w = np.exp(w_{p}) * w_{a}

# First we convert the anchors to have form [x, y, h, w] as well. Always remember here (x, y) stands for the center positon
anc_height = anchors[:, 2] - anchors[:, 0]
anc_width = anchors[:, 3] - anchors[:, 1]
anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

# Unparameterizing the predicted locations
dy, dx, dh, dw = torch.chunk(pred_anchor_locs.squeeze(), 4, 1)
pred_ctr_y = dy * anc_height.unsqueeze(1) + anc_ctr_y.unsqueeze(1)
pred_ctr_x = dx * anc_width.unsqueeze(1) + anc_ctr_x.unsqueeze(1)
pred_h = torch.exp(dh) * anc_height.unsqueeze(1)
pred_w = torch.exp(dw) * anc_width.unsqueeze(1)

# Getting roi boxes as [y1, x1, y2, x2]
roi = torch.stack([pred_ctr_y - 0.5 * pred_h, pred_ctr_x - 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h, pred_ctr_x + 0.5 * pred_w], 1).squeeze()
print(roi.shape)

# 2. Clipping the predicted boxes to the image
roi = torch.clamp(roi, 0, image.shape[2])
print(roi)

# 3. Removing predicted boxes that are too small
hs = roi[:, 2] - roi[:, 0]
ws = roi[:, 3] - roi[:, 1]
keep = (hs >= min_size) & (ws >= min_size)
print(keep)

roi = roi[keep!=0, :]
score = objectness_score.squeeze()[keep!=0]
print(roi.shape)
print(score.shape)

# 4. Sorting (proposal, score) pairs in descending order of score
order = torch.argsort(score.contiguous(), descending=True)
print(order)

# 5. Take top pre_nms_topN
order = order[:n_train_pre_nms]
roi = roi[order, :]
print(roi.shape)
print(roi)

# 6. Applying Non Maximal Suppression
y1 = roi[:, 0]
x1 = roi[:, 1]
y2 = roi[:, 2]
x2 = roi[:, 3]

area = (x2 - x1 + 1) * (y2 - y1 + 1)

score2 = score[order]
order2 = torch.argsort(score2.contiguous(), descending=True)

keep = []
while order2.size(0) > 0:
    # Choose the highest score roi
    i = order2[0]
    # add it to the keep list
    keep.append(i)

    # Find intersecting area
    xx1 = torch.max(x1[i], x1[order2[1:]])
    yy1 = torch.max(y1[i], y1[order2[1:]])
    xx2 = torch.min(x2[i], x2[order2[1:]])
    yy2 = torch.min(y2[i], y2[order2[1:]])

    w = torch.max(torch.Tensor([0.0]).to(device), xx2 - xx1 + 1)
    h = torch.max(torch.Tensor([0.0]).to(device), yy2 - yy1 + 1)
    inter = w * h

    # if intersecting area is beyond some nms threshold, do NOT add to the keep array
    overlap = inter / (area[i] + area[order2[1:]] - inter)

    inds = (overlap <= nms_thresh)

    # Delete entry from order if overlap is over nms threshold
    order2 = order2[1:]
    order2 = order2[inds == 1]

# Get final 2000 best nmsed rois
keep = keep[:n_train_post_nms]
roi = roi[keep]
print(roi)
print(roi.shape)

"""
Fast R-CNN

Thus far we have described the architecture and how to train(labels) a network for region proposal generation. 
For the detection network that utilizes the proposals from that network to do the final region-based object detection, we'll use Fast R-CNN

The Fast R-CNN network takes in the region proposals from the RPN, ground truth box locations, and their respective labels as inputs.

We will now generate labels for Fast R-CNN, and define it's structure.

Proposal Targets
"""
# Parameters used to generate the labels for the proposals

n_sample = 128           # Number of samples to sample from roi
pos_ratio = 0.25         # Number of positive examples out of the n_samples
pos_iou_thresh = 0.5     # Minimum overlap of region proposal with any ground truth object for it to be positive
neg_iou_thresh_hi = 0.5  # These are the range of IoU for it to be considered negative
neg_iou_thresh_lo = 0.0

# The proposal labels are basically calculated with their IoU with the ground truth
# The labelling process is almost exactly the same as the labelling process for RPN

# First we find the IoU of each ground truth object with the region proposals
# The following is the same code used above to calculate anchor box IoUs (RPN : ground truth <-> anchor / Fast R-CNN : ground_truth <-> proposals)

# First we get each coordinates as vectors separately
ya1, xa1, ya2, xa2 = torch.chunk(roi, 4, dim=1)  # Proposal Boxes
yb1, xb1, yb2, xb2 = torch.chunk(bbox, 4, dim=1)  # Ground truth Boxes

# Then we check the intersection by testing maximum/minimum
inter_x1 = torch.max(xa1, torch.transpose(xb1, 0, 1))  # This will result in (n_roi, 2) Tensor, which has the maximum for each ground truth box vs proposal box
inter_y1 = torch.max(ya1, torch.transpose(yb1, 0, 1))
inter_x2 = torch.min(xa2, torch.transpose(xb2, 0, 1))
inter_y2 = torch.min(ya2, torch.transpose(yb2, 0, 1))

# Calculating the intersection
inter_area = torch.max((inter_y2 - inter_y1 + 1), torch.Tensor([0, 0]).to(device)) * torch.max(
    (inter_x2 - inter_x1 + 1), torch.Tensor([0, 0]).to(device))
# The multiplication above is not matrix multiplication (the dimensions don't match anyway), it's scalar-wise multiplication
# The output has shape of (8940, 2), each column representing the area of intersection of each anchor box with corresponding bounding box (1st column = 1st bounding box) and so on

# Calculating the Union
anchor_area = (xa2 - xa1 + 1) * (ya2 - ya1 + 1)  # (8940, 1)
bbox_area = (xb2 - xb1 + 1) * (yb2 - yb1 + 1)  # (2, 1)

union_area = anchor_area + torch.transpose(bbox_area, 0, 1) - inter_area

# Calculating the IoU
proposal_gt_iou = inter_area / union_area
print(proposal_gt_iou.shape)  # blog got [1535, 2]
print(proposal_gt_iou)

# Find out which ground truth has the highest IoU for each proposal region, and the maximum IoU for each ground truth
gt_assignment = proposal_gt_iou.argmax(dim=1)
max_iou = proposal_gt_iou.max(dim=1)
print(gt_assignment, max_iou)

# Assigning the classification labels (ground truth labels) to each proposal
gt_roi_label = labels[gt_assignment]

# Now we need to select the foreground rois with the pos_iou_thresh parameter
pos_index = (max_iou[0] >= pos_iou_thresh).nonzero().transpose(0, 1)[0]
print(pos_index)

# We only want a maximum of 128x0.25 = 32 foreground samples
pos_roi_per_this_image = n_sample * pos_ratio
pos_roi_per_this_image = int(min(pos_roi_per_this_image, pos_index.size(0)))
if pos_index.size(0) > 0:
    pos_index = reservoir_sample(pos_index, pos_roi_per_this_image).long()

print(pos_roi_per_this_image)
print(pos_index)

# We do the same for the negative region proposals too
neg_index = ((neg_iou_thresh_lo <= max_iou[0]) & (max_iou[0] < neg_iou_thresh_hi)).nonzero().transpose(0, 1)[0]
print(neg_index)

# We only want a maximum of 128x0.25 = 32 foreground samples
neg_roi_per_this_image = n_sample - pos_roi_per_this_image
neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size(0)))

if neg_index.size(0) > 0:
    neg_index = reservoir_sample(neg_index, neg_roi_per_this_image).long()

print(neg_roi_per_this_image)
print(neg_index)

# Collecting both positive and negative samples
keep_index = torch.cat((pos_index, neg_index)).long()
print(keep_index)

gt_roi_labels = gt_roi_label[keep_index].to(device)
gt_roi_labels[pos_roi_per_this_image:] = 0  # negative labels --> 0
gt_roi_labels.to(device)
sample_roi = roi[keep_index]
print(sample_roi.shape)
print(gt_roi_labels)

bbox_for_sampled_roi = bbox[gt_assignment[keep_index]]
bbox_for_sampled_roi.to(device)
print(bbox_for_sampled_roi.shape)

# We use the ground truth objects for these sample_roi and paramterize it as we have done with the anchor boxes
height = sample_roi[:, 2] - sample_roi[:, 0]
width = sample_roi[:, 3] - sample_roi[:, 1]
ctr_y = sample_roi[:, 0] + 0.5 * height
ctr_x = sample_roi[:, 1] + 0.5 * width

base_height = bbox_for_sampled_roi[:, 2] - bbox_for_sampled_roi[:, 0]
base_width = bbox_for_sampled_roi[:, 3] - bbox_for_sampled_roi[:, 1]
base_ctr_y = bbox_for_sampled_roi[:, 0] + 0.5 * base_height
base_ctr_x = bbox_for_sampled_roi[:, 1] + 0.5 * base_width

# Parameterization
# Insure height and width are > 0
eps = 0.00001
height = torch.max(height, torch.full_like(height, eps))   # Ensure height and width are > 0
width = torch.max(width, torch.full_like(width, eps))

dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = torch.log(base_height / height)
dw = torch.log(base_width / width)

gt_roi_locs = torch.stack((dy, dx, dh, dw), 1)

"""
We have finally computed the gt_roi_locs and the gt_roi_labels for the sampled rois (sampled from all the proposals)

gt_roi_locs are the targets for the final regression of bounding box coordinates of the object
gt_roi_labels are the targets for the final classification of each bounding box into it's object class [6 or 8]
Now that we have computed all the targets requried for the Fast R-CNN part, we can finish this whole project off by

Defining the Fast R-CNN Network
Get a prediciton
Compute the Fast R-CNN loss
Compute the Total Loss
"""

"""
Fast R-CNN Architecture

Fast R-CNN uses ROI Pooling to extract features from each and every proposal suggested by the RPN. 

ROI Pooling basically is a max pooling operation on inputs of non uniform sizes to obtain a fixed size feature map.
It takes in 2 inputs

- A fixed sized feature map obtained from the feature extractor
- A matrix representing a list of our RoIs with shape (N,5), N would be the number of our rois, 5 for 1 index + 4 coordinates
"""
# So first we shall define our inputs, we've already got the fixed size (50x50) feature map from the feature extractor, so we just have to prepare the list of RoIs

# samples_roi is the list of our final sampled region proposals, but it has shape (128, 4) all we need to do is to add an index part to make it (128,5)
# and changing the coordinate system from [y1, x1, y2, x2] to [x center, y center, width, height]

# First we create a tensor of indices
roi_indicies = torch.zeros(sample_roi.size(0)).to(device)

# Then we concatenate the tensor into our sample_roi tensor
idx_and_rois = torch.cat((roi_indicies[:, None], sample_roi), dim=1)

# Reordering the tensor to have (index, x, y, h, w)
xy_idx_and_rois = idx_and_rois[:, [0, 2, 1, 4, 3]]
idx_and_rois = xy_idx_and_rois.contiguous()

print(sample_roi[0:3])
print(idx_and_rois[0:3])

# Now that we got both the feature map, and the region proposals,
# We can feed it to the roi pooling layer!

# We want a fixed feature map size from various different inputs,
# Let's choose the fixed size to be 7x7 and we use adaptive max poooling to deal with the variable size

size = (7,7)
adaptive_max_pool = nn.AdaptiveMaxPool2d((size[0], size[1])).to(device)

# Empty output array
output = []

# Take only the box coordinates (not the index)
rois = idx_and_rois.data.float()
# Scale it to our sub sampling ratio so that it fits the feature map size
rois[:, 1:].mul_(1/sub_sample)
rois = rois.long()

num_rois = rois.size(0)

# Iterate over each proposed region
for i in range(num_rois):
    roi = rois[i]
    # index of roi
    im_idx = roi[0]
    # Tbh, the first dimension is just the number of images, im_idx is 0, so .narrow(0,0,1) in this case is the same but okay
    # Maybe it has the structure for future improvements in batch training
    # We get the roi from the feature map
    # ... (however many : we need ), and we just index the last 2 part which are the x,y spatial dimensions
    im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
    output.append(adaptive_max_pool(im))

output = torch.cat(output)
print(output.size())

# And we reshape the pooled tensors so that we can feed it to a linear layer
k = output.view(output.size(0), -1)
print(k.shape)

# These pooled roi feature maps now go through a linear layer, then branching out to two output layers, the classification of the image, and the regression of bounding boxes
roi_head_layer = nn.Sequential(*[nn.Linear(25088, 4096), nn.Linear(4096,4096)]).to(device)
cls_loc = nn.Linear(4096, 21*4).to(device)  # VOC 20 classes + 1 background, each with 4 coordinates

cls_loc.weight.data.normal_(0, 0.01)
cls_loc.bias.data.zero_()

score = nn.Linear(4096, 21).to(device)

# passing the output k in to the network

k = roi_head_layer(k)
roi_cls_loc = cls_loc(k)
roi_cls_score = score(k)
print(roi_cls_loc.shape, roi_cls_score.shape)

# AND WE'RE DONE!! WOOHOO!
"""
Loss Functions

Now that we've defined the structure, targets and their predictions of both the

Region Proposal Network and the
Fast R-CNN Network
We can compute their respective loss functions, and thus move on to training the network

Recall that each network had two outputs:
The Region Proposal Network had an 'objectness' binary classification output and a regression output which predicted the object location
The Fast R-CNN Network has a multiclass classification output and a regression output which predicts the bounding box locations for each object
"""
# Recall our anchor targets and anchor predictions

print(pred_anchor_locs.shape)
print(pred_cls_scores.shape)
print(anchor_locations.shape)
print(anchor_label.shape)

# We rearrange the tensors to align the input and output
rpn_loc = pred_anchor_locs[0]
rpn_score = pred_cls_scores[0]   # Predicted Objectness Score

gt_rpn_loc = anchor_locations
gt_rpn_score = anchor_label      # Actual Objectness

# Calculating the classification loss first

rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_score.long(), ignore_index = -1)
print(rpn_cls_loss)

# Calculating the regression loss

# First dealing with the kronecker delta like parameter,  we only want the positive anchors
pos = gt_rpn_score > 0
mask = pos.unsqueeze(1).expand_as(rpn_loc)
print(mask.shape)

# Masking the rpn_location predictions and the gt_rpn_location targets with only the positive anchors
mask_loc_preds = rpn_loc[mask].view(-1, 4)
mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)
print(mask_loc_preds.shape, mask_loc_preds.shape)

# The loss
x = torch.abs(mask_loc_targets - mask_loc_preds)
rpn_loc_loss = ((x < 1).float() *0.5 * x**2) + ((x>=1).float()*(x-0.5))
print(rpn_loc_loss.sum())

# Combining the classification loss and the regression loss with a weight $\lambda$
# Classification is on all bounding boxes, but regression is only on positive boxes, so the weight is applied

rpn_lambda = 10
N_reg = (gt_rpn_score >0).float().sum()
rpn_loc_loss = rpn_loc_loss.sum() / N_reg
rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)
print(rpn_loss)

# And that is our RPN loss!!

"""
FAST RCNN LOSS
"""
# Again, recall that we've got the following predictions and targets from our Fast RCNN network

print(roi_cls_loc.shape)   # 4 values for 21 classes
print(roi_cls_score.shape)

print(gt_roi_locs.shape)
print(gt_roi_labels.shape)

# Classification loss
roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_labels, ignore_index=-1)
print(roi_cls_loss)

# Regression loss
n_sample = roi_cls_loc.shape[0]   # 128
roi_loc = roi_cls_loc.view(n_sample, -1, 4)   # reshape to (128, 21, 4)
print(roi_loc.shape)

roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_labels]  # For each roi, get the predicted box for the ground truth label -> and therefore, for each roi 1 roi will come out
print(roi_loc.shape)  # so this is going to be (128, 4)

# The loss
pos = gt_roi_labels > 0
mask = pos.unsqueeze(1).expand_as(roi_loc)
print(mask.shape)

# Masking the rpn_location predictions and the gt_rpn_location targets with only the positive anchors
mask_loc_preds = roi_loc[mask].view(-1, 4)
mask_loc_targets = gt_roi_locs[mask].view(-1, 4)
print(mask_loc_preds.shape, mask_loc_preds.shape)

x = torch.abs(mask_loc_targets - mask_loc_preds)
roi_loc_loss = ((x < 1).float() *0.5 * x**2) + ((x>=1).float()*(x-0.5))
print(roi_loc_loss.sum())

# Total ROI Loss
roi_lambda = 10.
roi_loss = roi_cls_loss + (roi_lambda * roi_loc_loss.sum())
print(roi_loss)

"""
TOTAL LOSS
"""
total_loss = rpn_loss + roi_loss

print(total_loss)

time2 = time.time()
duration = time2-time1
print(duration)
