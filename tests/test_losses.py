import pytest
import torch
from utils.losses import YoloV1Loss

def test_categorical_loss():
    #
    gt = torch.zeros(1,7,7,15)
    gt[0,0,0,4] = 1
    gt[0,0,0,6] = 1
    predicted = torch.zeros(1,7*7*20)
    predicted[0,4] = 1
    predicted[0,11] = 1

    loss = YoloV1Loss(7, 2, 10)
    print('Running Identical Class Test...')
    identical_class_loss = loss(predicted, gt)
    # Test when the classes are identical
    assert identical_class_loss == 0, f'Identical Class codes result in nonzero loss: {identical_class_loss}'
    print("Identical Class Test Passed")
    
    predicted[0,11] = 0
    predicted[0,12] = 1

    print('Running Separate Class Test...')
    separate_class_loss = loss(predicted, gt)
    # Test when the classes are different
    assert separate_class_loss == 2., f'Separate Loss should be 2.0, actual: {separate_class_loss}'
    print('Separate Class Test Passed')

    predicted[0,11] = 0.5
    predicted[0,12] = 0.5
    print('Running Split Class Test...')
    split_class_loss = loss(predicted, gt)
    print(split_class_loss)
    # Test when the prediction is split
    assert split_class_loss == .5, f'Split Loss should be 0.5, actual: {split_class_loss}'
    print('Split Class Test Passed')

    gt[0,0,0,4] = 0
    predicted[0,4] = 0
    print('Running No Object Test...')
    noobj_class_loss = loss(predicted, gt)
    # test when no object is present.
    assert noobj_class_loss == 0., f'No Object Loss should be 0., actual: {noobj_class_loss}'
    print('No Object Test Passed')

def test_detection_loss():
    gt = torch.zeros(1,7,7,15)
    gt[0,0,0,4] = 1
    gt[0,0,0,0] = 0.1
    gt[0,0,0,1] = 0.1
    gt[0,0,0,2] = 0.09
    gt[0,0,0,3] = 0.09

    predicted = torch.zeros(1,7*7*20)
    predicted[0,4] = 1
    predicted[0,0] = 0.1
    predicted[0,1] = 0.1
    predicted[0,2] = 0.09
    predicted[0,3] = 0.09
    loss = YoloV1Loss(7, 2, 10)

    print('Running Identical Detection Test...')
    identical_det_loss = loss(predicted, gt)
    assert identical_det_loss == 0, f'Identical Det Loss resulted in nonzero loss: {identical_det_loss}'
    print('Identical Detection Test Passed')

    print('Running Missed Detection Test...')
    predicted[0,4] = 0
    missed_det_loss = loss(predicted, gt)
    assert missed_det_loss == 1, f'Missed Det loss should be 1, actual: {missed_det_loss}'
    print('Missed Detection Test Passed')

    predicted[0,4] = 0
    predicted[0,160] = 0.1
    predicted[0,161] = 0.1
    predicted[0,162] = 0.09
    predicted[0,163] = 0.09
    predicted[0,164] = 1

    print('Running Wrong Location Test...')
    wrong_loc_loss = loss(predicted, gt)
    assert wrong_loc_loss == 1.5, f'Wrong Location loss should be 1.5, actual: {wrong_loc_loss}'
    print('Wrong Location Test Passed...')

def test_bbox_loss():
    gt = torch.zeros(1,7,7,15)
    gt[0,0,0,4] = 1
    gt[0,0,0,0] = 0.1
    gt[0,0,0,1] = 0.1
    gt[0,0,0,2] = 0.09
    gt[0,0,0,3] = 0.09
    predicted = torch.zeros(1,7*7*20)
    predicted[0,4] = 1
    predicted[0,0] = 1.1
    predicted[0,1] = -0.9
    predicted[0,2] = 0.25
    predicted[0,3] = 0.25
    loss = YoloV1Loss(7, 2, 10)
    bbox_loss = loss(predicted, gt)
    assert torch.abs(bbox_loss - 5*2.08) < 1e-10, f'BBox Loss should be 2.08: {bbox_loss}'
