import math
import sys
import time
import numpy as np

  
def overlap(x1_1,y1_1,x2_1,y2_1,x1_2,y1_2,x2_2,y2_2):
  w = min(x2_1,x2_2) - max(x1_1,x1_2)
  h = min(y2_1,y2_2) - max(y1_1,y1_2)
  
  if w<=0. or h<=0.:
    return 0.
  return w*h/((x2_1-x1_1)*(y2_1-y1_1)+(x2_2-x1_2)*(y2_2-y1_2)-w*h)

# anchors: x,y,w,h  deltas: deltay,deltax,deltah,deltaw classes: 91 need sigmoid
def get_results(anchors,deltas,classes,imw,imh,conf_thresh,iou_thresh,max_num):
  an_i,an_j = anchors.shape
  de_i,de_j = deltas.shape
  cl_i,cl_j = classes.shape
  assert an_i == de_i
  assert an_i == cl_i
  
  class_sig = 1./(1.+np.exp(-classes))
  
  class_ids = []
  
  for i in range(an_i):
    label_prob = []
    max_prob = 0
    max_label = 0
    for j in range(cl_j):
      if class_sig[i][j] > max_prob:
        max_prob = class_sig[i][j]
        max_label = j
    label_prob.append(max_label)
    label_prob.append(max_prob)
    class_ids.append(label_prob)
  
  before_box_num = 0  
  before_boxes = []
  to_get_sort = []  # save probs to sort
  for i in range(an_i):
    if class_ids[i][1] > conf_thresh:
      before_box_num = before_box_num + 1
      new_box = []
      to_get_sort.append(class_ids[i][1])
      new_box.append(class_ids[i][0])
      new_box.append(class_ids[i][1])
      
      deltaY = 0.1 * deltas[i][0]
      deltaX = 0.1 * deltas[i][1]
      deltaH = 0.2 * deltas[i][2]
      deltaW = 0.2 * deltas[i][3]
      oriX = anchors[i][0]
      oriY = anchors[i][1]
      oriW = anchors[i][2]
      oriH = anchors[i][3]
      
      newY = oriY + deltaY * oriH
      newX = oriX + deltaX * oriW
      newH = oriH * math.exp(deltaH)
      newW = oriW * math.exp(deltaW)
      
      # y1 x1 y2 x2
      y1 = min(max((newY-newH/2)/imh,0.),1.)
      x1 = min(max((newX-newW/2)/imw,0.),1.)
      y2 = min(max((newY+newH/2)/imh,0.),1.)
      x2 = min(max((newX+newW/2)/imw,0.),1.)
      
      new_box.append(y1)
      new_box.append(x1)
      new_box.append(y2)
      new_box.append(x2)
      new_box.append(1)
      
      before_boxes.append(new_box)
      
  # before boxes  clsid prob y1 x1 y2 x2 ifretain
  #iou_thresh = 0.6
  #max_num = 100
  to_get_sort = np.array(to_get_sort)
  sort_id = np.argsort(-to_get_sort)
  
  results = []
  for i in range(before_box_num):
    if before_boxes[sort_id[i]][6] == 1:
      results.append(before_boxes[sort_id[i]])
      
      base_x1 = before_boxes[sort_id[i]][3]
      base_y1 = before_boxes[sort_id[i]][2]
      base_x2 = before_boxes[sort_id[i]][5]
      base_y2 = before_boxes[sort_id[i]][4]
      
      for j in range(i+1,before_box_num):
        if before_boxes[sort_id[i]][0] != before_boxes[sort_id[j]][0]:
          continue
        if before_boxes[sort_id[i]][6] == 0:
          continue
        cur_x1 = before_boxes[sort_id[j]][3]
        cur_y1 = before_boxes[sort_id[j]][2]
        cur_x2 = before_boxes[sort_id[j]][5]
        cur_y2 = before_boxes[sort_id[j]][4]
        
        if overlap(base_x1,base_y1,base_x2,base_y2,cur_x1,cur_y1,cur_x2,cur_y2) > iou_thresh:
          before_boxes[sort_id[j]][6] = 0
  
  return results
        

def get_anchor_list(levels,img_side,scales,aspect_ratios,offset_ratio,anchor_scale):
	assert len(scales) == len(aspect_ratios)

	ret = []

	for level in levels:
		stride = math.pow(2,level)
		if img_side % stride != 0:
			print('img_side stride: ',stride)
			sys.exit(-1)
		offset = offset_ratio * stride
		base_anchor_side = anchor_scale * stride
		base_anchors = []
		for scale,aspect in zip(scales,aspect_ratios):
			aspect = math.sqrt(aspect)
			base_anchors.append([offset,offset,base_anchor_side*scale*aspect,base_anchor_side*scale/aspect])
		feat_side = int(img_side / stride)
		for y in range(feat_side):
			for x in range(feat_side):
				for b_anchor in base_anchors:
					ret.append([b_anchor[0]+x*stride,b_anchor[1]+y*stride,b_anchor[2],b_anchor[3]])

	return ret


if __name__=="__main__":
  levels = [3,4,5,6,7]
  img_side = 640
  scales = [1,math.pow(2.,(1./2.)),1,math.pow(2.,(1./2.)),1,math.pow(2.,(1./2.))]
  print(scales)
  aspect_ratios = [1,1,2,2,0.5,0.5]
  offset_ratio = 0.5
  anchor_scale = 4
  
  loop = 100
  
  start = time.clock()
  for i in range(loop):
  	anchors = get_anchor_list(levels,img_side,scales,aspect_ratios,offset_ratio,anchor_scale)
  end = time.clock()
  print(len(anchors))
  print(anchors[51073])
  print('time: ',(end-start)/loop)
