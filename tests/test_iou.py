import pytest
import torch
from utils.misc import intersection_over_union

# Test if iou of empty bounding boxes is 0
def test_full_empty_bbox_iou():
    empty_bbox = torch.zeros([1,7,7,4])
    max_iou = torch.max(intersection_over_union(empty_bbox, empty_bbox))

    assert max_iou == 0, 'IoU of Empty BBoxes != 0'

# Test if only one bbox is empty
def test_partial_empty_bbox_iou():
    empty_bbox = torch.zeros([1,7,7,4])
    bbox_a = torch.zeros([1,7,7,4])
    bbox_a[:,3,4,0] = 0.1
    bbox_a[:,3,4,1] = 0.1
    bbox_a[:,3,4,2] = 0.1
    bbox_a[:,3,4,3] = 0.1

    max_iou = torch.max(intersection_over_union(empty_bbox, empty_bbox))
    assert  max_iou == 0, 'IoU of Empty BBox w/ non-empty != 0'

# Test non-overlapping bboxes
def test_disjoint_bbox_iou():
    bbox_a = torch.zeros([1,7,7,4])
    bbox_a[:,3,4,0] = 0.1
    bbox_a[:,3,4,1] = 0.1
    bbox_a[:,3,4,2] = 0.1
    bbox_a[:,3,4,3] = 0.1

    bbox_b = torch.zeros([1,7,7,4])
    bbox_b[:,3,4,0] = 0.25
    bbox_b[:,3,4,1] = 0.25
    bbox_b[:,3,4,2] = 0.1
    bbox_b[:,3,4,3] = 0.1

    max_iou = torch.max(intersection_over_union(bbox_a, bbox_b))
    assert max_iou == 0

# Test identical bboxes
def test_ident_bbox_iou():
    bbox_a = torch.zeros([1,7,7,4])
    bbox_a[:,3,4,0] = 0.1
    bbox_a[:,3,4,1] = 0.1
    bbox_a[:,3,4,2] = 0.1
    bbox_a[:,3,4,3] = 0.1
    
    max_iou = torch.max(intersection_over_union(bbox_a, bbox_a))
    assert max_iou - 0.9999000099990002 <=  1e-16

    # Test partially overlapped bboxes
def test_overlap_bbox_iou():
    bbox_a = torch.zeros([1,7,7,4])
    bbox_a[:,3,4,0] = 0.1
    bbox_a[:,3,4,1] = 0.1
    bbox_a[:,3,4,2] = 0.6
    bbox_a[:,3,4,3] = 0.6

    bbox_b = torch.zeros([1,7,7,4])
    bbox_b[:,3,4,0] = 0.2
    bbox_b[:,3,4,1] = 0.2
    bbox_b[:,3,4,2] = 0.6
    bbox_b[:,3,4,3] = 0.6

    iou = intersection_over_union(bbox_a, bbox_b)
    target = torch.zeros(1,7,7,1)
    target[0,3,4,0] = .5**2 / (.6**2 + .6**2 - .5**2 + 1e-6)
    max_iou = torch.max(torch.abs(iou - target))
    assert max_iou <  1e-6, f'Target: {target[0,3,4,0]}. IoU: {iou[0,3,4,0]}, diff: {max_iou}'
     
# Test one bbox inside another
def test_contained_bbox_iou():
    bbox_a = torch.zeros([1,7,7,4])
    bbox_a[:,3,4,0] = 0.1
    bbox_a[:,3,4,1] = 0.1
    bbox_a[:,3,4,2] = 1.0
    bbox_a[:,3,4,3] = 1.0

    bbox_b = torch.zeros([1,7,7,4])
    bbox_b[:,3,4,0] = 0.2
    bbox_b[:,3,4,1] = 0.2
    bbox_b[:,3,4,2] = 0.5
    bbox_b[:,3,4,3] = 0.5

    iou = intersection_over_union(bbox_a, bbox_b)
    target = torch.zeros(1,7,7,1)
    target[0,3,4,0] = .5**2 / (.5**2 + 1**2 - .5**2 + 1e-6)
    max_iou = torch.max(torch.abs(iou - target))
    assert max_iou <  1e-6, f'Target: {target[0,3,4,0]}. IoU: {iou[0,3,4,0]}, diff: {max_iou}'
    
# Test 1x1 grid bbox
def test_1x1_bbox_iou():
    bbox_a = torch.zeros([1,1,1,4])
    bbox_a[:,0,0,0] = 0.1
    bbox_a[:,0,0,1] = 0.1
    bbox_a[:,0,0,2] = 1.0
    bbox_a[:,0,0,3] = 1.0

    bbox_b = torch.zeros([1,1,1,4])
    bbox_b[:,0,0,0] = 0.2
    bbox_b[:,0,0,1] = 0.2
    bbox_b[:,0,0,2] = 0.5
    bbox_b[:,0,0,3] = 0.5

    iou = intersection_over_union(bbox_a, bbox_b)
    target = torch.zeros(1,1,1,1)
    target[0,0,0,0] = .5**2 / (.5**2 + 1**2 - .5**2 + 1e-6)
    max_iou = torch.max(torch.abs(iou - target))
    assert max_iou <  1e-6, f'Target: {target[0,0,0,0]}. IoU: {iou[0,0,0,0]}, diff: {max_iou}'
    
# Test bbox with categorical info
def test_bbox_w_cat_iou():
    bbox_a = torch.zeros([1,7,7,25])
    bbox_a[:,0,0,0] = 0.1
    bbox_a[:,0,0,1] = 0.1
    bbox_a[:,0,0,2] = 1.0
    bbox_a[:,0,0,3] = 1.0

    bbox_b = torch.zeros([1,7,7,25])
    bbox_b[:,0,0,0] = 0.2
    bbox_b[:,0,0,1] = 0.2
    bbox_b[:,0,0,2] = 0.5
    bbox_b[:,0,0,3] = 0.5

    iou = intersection_over_union(bbox_a, bbox_b)
    target = torch.zeros(1,7,7,1)
    target[0,0,0,0] = .5**2 / (.5**2 + 1**2 - .5**2 + 1e-6)
    max_iou = torch.max(torch.abs(iou - target))
    assert max_iou <  1e-6, f'Target: {target[0,0,0,0]}. IoU: {iou[0,0,0,0]}, diff: {max_iou}'
    