import glob
import json
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from craft_text_detector import Craft
from PIL import Image, ImageDraw, ImageFont
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from tqdm import tqdm
from default import boundingRect
config = Cfg.load_config_from_name('vgg_seq2seq')
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False
vietocr = Predictor(config)

craft = Craft(crop_type='box', cuda=True,)

def rulebased(image_path):
    src = cv2.imread(image_path, cv2.IMREAD_COLOR)

    """
    Modify from : https://docs.opencv.org/4.1.2/dd/dd7/tutorial_morph_lines_detection.html
    Add binary result and  
    """

    # Check if image is loaded fine
    assert src is not None

    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv2.bitwise_not(gray)
    # bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #                             cv2.THRESH_BINARY, 15, -5)
    _, bw = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    # Create the images that will use to extract the horizontal and vertical lines
    vertical = np.copy(bw)


    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 20
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    find_cnt = vertical
    # Inverse vertical image
    vertical = cv2.bitwise_not(vertical)
    horizontal = np.copy(bw)
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 20
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=1)
    horizontal = cv2.bitwise_not(horizontal)


    result = cv2.bitwise_and(horizontal, vertical)

    #Finally, apply binary threshold to result
    result = cv2.threshold(result, 225, 255, cv2.THRESH_BINARY)[1]
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(result, connectivity=4, ltype=cv2.CV_32S)

    dict_of_cols = {}
    key_matching = {}
    dict_of_OCR = {}
    dict_of_det = {}
    output = src.copy()
    list_detect = []
    square = src.shape[0] * src.shape[1]
    for i in range(len(stats)):
        x, y, w, h, area = stats[i]
        # print(x, y, w, h, area)
        if w*h > 0.6*square:
            continue
        # OCR boxes
        img_box = output[y:y+h, x:x+w]
        try:
            text_boxes = craft.detect_text(image=img_box)['boxes']
        except:
            continue
        if len(text_boxes) > 0:
            text_boxes = [boundingRect(bbox, img_box) for bbox in text_boxes]
            text_boxes = sorted(text_boxes, key=lambda x: (x[1], x[0]))
        s = []
        legit_box = []
        for bbox in text_boxes:
            x0, y0, x1, y1 = list(map(int, bbox))
            if x0 < 0 or y0 < 0 or y1 - y0 > 60:
                continue
            try:
                img = img_box[y0:y1,x0:x1]
                img = Image.fromarray(img)
            except:
                print("ERROR", image_path)
                continue
            list_detect.append(img)
            pred = vietocr.predict(img)
            if pred:
                s.append(pred)
                bbox = (x + x0, y + y0, x + x1, y + y1)
                dict_of_det[bbox] = (pred, i)
        if s:
            dict_of_OCR[i] = " ".join(s)

    offset = 5
    number_cols = {}
    for key in dict_of_OCR.keys():
        if not dict_of_OCR[key]:
            continue
        x, y, w, h, _ = stats[key]
        # Match rows
        xc, yc = centroids[key]
        cols = 0
        for i in range(len(stats)):
            _, _, wk, hk, _ = stats[i]
            xck, yck = centroids[i]
            if abs(yck - yc) < offset and abs(hk - h) < offset:
                    cols += 1
        number_cols[key] = cols

    from collections import defaultdict
    key_matching = defaultdict(list)
    dict_of_cols = {}
    dict_of_rows = {}
    offset_x = 5
    offset_y = 80
    max_col = max(number_cols.values())

    annotations = []

    for i in dict_of_OCR.keys():
        if not dict_of_OCR[i]:
            continue
        x, y, w, h, _ = stats[i]
        xc, yc = centroids[i]
        new_row = True
        new_col = True        
        label = "Other"
        # Matching by row
        if number_cols[i] < max_col:
            for key in dict_of_rows.keys():
                _, _, wk, hk, _ = stats[i]
                xck, yck = centroids[key]
                if abs(yck - yc) < offset_x and abs(hk - h) < offset_x:
                    new_col = False
                    dict_of_rows[key].append(i)
                    # key_matching[dict_of_OCR[key]].append(dict_of_OCR[i])
                    if"Total" in dict_of_OCR[key] and len(dict_of_rows[key]) > 2:
                        label = "Total price"
            if new_col:
                dict_of_rows[i] = []
                # key_matching[dict_of_OCR[i]] = []
            key_matching[label].append((dict_of_OCR[i], i))
            continue

        # Matching by column
        for key in dict_of_cols.keys():
            _, _, wk, hk, _ = stats[i]
            xck, yck = centroids[key]
            if abs(xck - xc) < offset_x and abs(wk - w) < offset_x and number_cols[i] == number_cols[key] :
                new_row = False
                dict_of_cols[key].append(i)
                # key_matching[dict_of_OCR[key]].append(dict_of_OCR[i])
                if "(Description)" in dict_of_OCR[key] and len(dict_of_cols[key]) > 0:
                    label = "Item"
                elif "(Amount)" in dict_of_OCR[key] and len(dict_of_cols[key]) > 0:
                    label = "Price"
                elif "Total" in dict_of_OCR[key] and len(dict_of_cols[key]) > 0:
                    label = "Total price"
                break
                break

        if new_row:
            dict_of_cols[i] = []

        key_matching[label].append((dict_of_OCR[i], i))

    annotation = []
    i = 0
    for key, value in dict_of_det.items():
        key = list(map(int, key))
        text, id = value[0], value[1]
        for label in key_matching.keys():
            for matched in key_matching[label]:
                text_compare, id_compare = matched[0], matched[1]
                if text in text_compare and id == id_compare:
                    hahaha = label
        
        annotation.append(dict(
            id=i,
            text=text,
            label=hahaha,
            box=key,
            linking=[]
        ))
        i += 1

    return annotation

if __name__ == "__main__":
    img_paths = glob.glob("/mnt/disk2/viethd/data/Grab/*.jpg")
    print("Number of receipt images is:", len(img_paths))
    for path in tqdm(img_paths):
        annotation = rulebased(path)
        basename = os.path.basename(path)[:-4]
        with open("/mnt/disk2/viethd/KIE-LABELED/" + basename + ".json", "w", encoding='utf8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=4) 
        shutil.copy(path, "/mnt/disk2/viethd/KIE-LABELED/")