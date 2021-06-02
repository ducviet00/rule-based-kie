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

config = Cfg.load_config_from_name('vgg_seq2seq')
config['device'] = 'cuda:0'
config['predictor']['beamsearch']=False
vietocr = Predictor(config)

craft = Craft(crop_type='box', cuda=True,)

def boundingRect(bbox, img):
    bbox = np.array(bbox)
    bbox = np.clip(bbox, 0, max(img.shape[:2]))
    x0, y0 = np.min(bbox, axis=0)
    x1, y1 = np.max(bbox, axis= 0)
    bbox = bbox.astype(np.int64)
    return x0, y0, x1, y1

def text_recog(img_path, visualize=False):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if visualize:
        im_pil = Image.new('RGB', (img.shape[1], img.shape[0]))
        draw = ImageDraw.Draw(im_pil)  
        font = ImageFont.truetype(r'/content/FiraSans-Regular.ttf', 24)

    annotations = []
    text_boxes = craft.detect_text(image=img)['boxes']

    if len(text_boxes) > 0:
        text_boxes = [boundingRect(bbox, img) for bbox in text_boxes]
        text_boxes = sorted(text_boxes, key=lambda x: (x[1], x[0]))

    for i, bbox in enumerate(text_boxes):
        x0, y0, x1, y1 = list(map(int, bbox))
        try:
            img_box = Image.fromarray(img[y0:y1, x0:x1])
        except:
            print(img.shape)
            print(list(map(int, bbox)))
        pred = vietocr.predict(img_box)
        if visualize:
            draw.text((x0, y0), pred, fill ="white", font = font, align ="right")
        annotations.append(dict(
                id=i,
                text=pred,
                label="Other",
                box=list(map(int, bbox)),
                linking=[]
            ))
    return annotations

if __name__ == "__main__":
    img_paths = glob.glob("/mnt/disk2/baohg/data/*/*.jpg")
    print("Number of receipt images is:", len(img_paths))
    for path in tqdm(img_paths):
        annotation = text_recog(path)
        basename = os.path.basename(path)[:-4]
        with open("/mnt/disk2/viethd/KIE/" + basename + ".json", "w", encoding='utf8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=4) 
        shutil.copy(path, "/mnt/disk2/viethd/KIE/")