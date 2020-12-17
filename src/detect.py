import SimpleITK as sitk
import numpy as np
import scipy
import math
import torch
import os
import radiomics.featureextractor


def HU2uint8(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    """
    Convert HU unit into uint8 values. First bound HU values by predfined min
    and max, and then normalize
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """
    image_new = np.array(image)
    image_new[np.isnan(image_new)] = HU_nan

    # normalize to [0, 1]
    image_new = (image_new - HU_min) / (HU_max - HU_min)
    image_new = np.clip(image_new, 0, 1)
    image_new = (image_new * 255).astype('uint8')

    return image_new


def pad2factor(image, factor=16, pad_value=0):
    depth, height, width = image.shape
    d = int(math.ceil(depth / float(factor))) * factor
    h = int(math.ceil(height / float(factor))) * factor
    w = int(math.ceil(width / float(factor))) * factor

    pad = [[0, d - depth], [0, h - height], [0, w - width]]

    image = np.pad(image, pad, 'constant', constant_values=pad_value)

    return image


def get_type(image=None, mask=None, classification_model=None):
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllImageTypes()
    fv = extractor.execute(image, mask)
    start_index = 22
    fv = np.array([v for k, v in fv.items()][start_index:])
    value = classification_model.predict(fv.reshape(1, -1)).squeeze()
    nodule_type = None
    if value <= 1.5:
        nodule_type = 1
    elif 1.5 < value <= 2.5:
        nodule_type = 2
    elif 2.5 < value < 3.5:
        nodule_type = 3
    elif 3.5 <= value < 4.5:
        nodule_type = 4
    elif value >= 4.5:
        nodule_type = 5

    return str(nodule_type)


def get_score(nodule_type, diameter, base_new_grow, calcification, spiculation,
              perifissural, endobronchial, preferences):
    diameter = eval(diameter)
    type_score = '0'
    special_score = '0'
    if calcification == '1':
        special_score = '1'
    if perifissural == '1' and diameter < 10:
        special_score = '2'
    if endobronchial == '1':
        special_score = '4A'
    if spiculation == '1':
        special_score = '4X'

    if not preferences['automatic_classification']:
        if nodule_type == '2' or nodule_type == '4':
            return '?'

    if nodule_type == '5' or nodule_type == '4':
        if base_new_grow == 0:
            if diameter < 6:
                type_score = '2'
            elif 6 <= diameter < 8:
                type_score = '3'
            elif 8 <= diameter < 15:
                type_score = '4A'
            elif 15 <= diameter:
                type_score = '4B'
        elif base_new_grow == 1:
            if diameter < 4:
                type_score = '2'
            elif 4 <= diameter < 6:
                type_score = '3'
            elif 6 <= diameter < 8:
                type_score = '4A'
            elif 8 <= diameter:
                type_score = '4B'

        elif base_new_grow == 2:
            if diameter < 8:
                type_score = '4A'
            elif 8 <= diameter:
                type_score = '4B'

    elif nodule_type == '3':
        # Rougthly set diameter of solid component of part-solid to 0.5 * diameter
        solid_part = 0.5 * diameter

        if solid_part >= 8:
            type_score = '4B'

        if base_new_grow == 0:
            if diameter < 6:
                type_score = '2'
            elif diameter >= 6:
                if solid_part < 6:
                    type_score = '3'
                elif 6 <= solid_part < 8:
                    type_score = '4A'

        elif base_new_grow == 1:
            if diameter < 6:
                type_score = '3'

    elif nodule_type == '1' or nodule_type == '2':
        if base_new_grow == 0:
            if 30 <= diameter:
                type_score = '3'
            elif diameter < 30:
                type_score = '2'
        elif base_new_grow == 1 or base_new_grow == 2:
            if 30 <= diameter:
                type_score = '3'

    for s in ['4X', '4B', '4A', '3', '2', '1', '0']:
        if special_score == s or type_score == s:
            return s


# TODO 3/4 more than 3 months remains the same can reduce it score
def get_base_new_grow(coord):
    base = 0
    new = 1
    grow = 2
    return base


def detect(filename, nodulenet_model=None, classification_model=None, preferences=None):
    original_image = sitk.GetArrayFromImage(sitk.ReadImage(filename))
    original_image = HU2uint8(original_image)

    temp_image = original_image[np.newaxis, ...]
    temp_image = pad2factor(temp_image[0])
    temp_image = np.expand_dims(temp_image, 0)
    input_image = (temp_image.astype(np.float32) - 128.) / 128.
    input_image = torch.from_numpy(input_image).float()
    del temp_image

    with torch.no_grad():
        input_image = input_image.unsqueeze(0)
        nodulenet_model.forward(input_image, None, None, None, None)

    detections = nodulenet_model.detections.cpu().numpy()
    mask_probs = np.asarray([t.cpu().numpy() for t in nodulenet_model.mask_probs], dtype=np.object)
    crop_boxes = nodulenet_model.crop_boxes

    threshold = float(preferences['threshold'])
    wanted = []
    for i, detection in enumerate(detections, start=0):
        if detection[1] > threshold:
            wanted.append(i)
    crop_boxes = crop_boxes[wanted]
    mask_probs = mask_probs[wanted]
    detections = detections[wanted]

    """
    Since NoduleNet detect nodule's longest axis in 3D as diameter, but Lung-RADS use mean of long and short axis 
    on axial view as diameter, roughly use a coefficient to approximate NoduleNet's diameter to Lung-RADS's diameter 
    """
    diameter_coef = 0.5
    csv = []
    for d, b, p in zip(detections, crop_boxes, mask_probs):
        # d[1] is prob, d[2,3,4] is x,y,z, d[5] is diameter
        diameter = f'{np.mean(d[5:]) * diameter_coef:.2f}'
        # type要把crop和mask也讀進來處理
        image = original_image[b[1]:b[4], b[2]:b[5], b[3]:b[6]]
        mask = np.zeros_like(image, dtype=int)
        mask[p > 0] = 1

        image = sitk.GetImageFromArray(image)
        mask = sitk.GetImageFromArray(mask)
        mask.CopyInformation(image)

        # TODO automatic classification of type beside Non-Solid/Part-Solid/Solid
        calcification, spiculation, perifissural, endobronchial = '0', '0', '0', '0'

        """
        For single patient, base nodule (first screening for this patient), new nodule, grown nodule have different 
        judgement criteria. To distinguish between them coordinate of nodule needs to ne analysis, but this haven't 
        be done yet.
        """
        # TODO base_new_grow
        base_new_grow = get_base_new_grow(coord=[])
        nodule_type = get_type(image, mask, classification_model)
        score = get_score(nodule_type, diameter, base_new_grow, calcification, spiculation, perifissural, endobronchial,
                          preferences)

        csv.append({
            "x": str(int(d[2])), "y": str(int(d[3])), "z": str(int(d[4])), "prob": str(round(d[1], 2)),
            "diameter": str(diameter), "type": nodule_type, "score": score,
            "calcification": calcification, "spiculation": spiculation, "perifissural": perifissural,
            "endobronchial": endobronchial
        })

    head, tail = os.path.split(filename)
    directory = f'result/{tail}'
    os.makedirs(directory, exist_ok=True)
    np.save(f'{directory}/detections.npy', detections)
    np.save(f'{directory}/crop_boxes.npy', crop_boxes)
    np.save(f'{directory}/mask_probs.npy', mask_probs)

    return csv
