
# coding: utf-8

# # About
# 
# TBD
import ipywidgets as widgets
from ipyevents import Event 
from IPython.display import display, clear_output, Image
import json
from io import BytesIO
from PIL import Image as PILImage
from pathlib import Path

int_slider = widgets.IntSlider(
    value=7,
    min=-1,
    max=9,
    step=1,
    description='Test:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)
int_slider

int_slider.value = 0

c = widgets.Label('Click or type on me!')
c.layout.border = '10px solid green'
d = Event(source=c, watched_events=['keydown'])
h = widgets.HTML('Event info')

def handle_event(event):
    lines = ['{}: {}'.format(k, v) for k, v in event.items()]
    content = '<br>'.join(lines)
    h.value = content
    key = event['key']
    try:
        n = int(key)
    except ValueError:
        n = -1
    int_slider.value = n

d.on_dom_event(handle_event)                  
display(c, h)

hhh = widgets.HTML('HI!')
hhh

hhh.value = 'test'

from time import sleep

i = Image.open(img)
i.wri

jpath = '/Users/gosuke/Downloads/via_project_27Feb2019_22h13m(1).json'
with open(jpath) as f:
    d = json.load(f)

for k in d.keys():
    print(k)
    if 'metadata' not in k:
        continue
    for k in d[k]:
        print(k)
        break
    break

k

d['_via_img_metadata'][k]

import string
string.ascii_lowercase

class LegoDB:
    

ial.update(pilimage2bytes(PILImage.open('test.png')), 'test_img!')

ial.update(pilimage2bytes(PILImage.open('test2.png')), 'test2_img!')


# # classes
# 
# hello
def pilimage2bytes(pil_image):
    with BytesIO() as f:
        pil_image.save(f, format='jpeg')
        f.seek(0)
        return f.read()

def parse_via_json(path):
    """Convert VII json to a list (file_name, file_size, x, y)"""
    
    with open(path) as f:
        d = json.load(f)
        
    d = d['_via_img_metadata']
    
    ret = []
    
    for fname_size in d.keys():
        i = fname_size.lower().index('.jpg')
        i = i + len('.jpg')
        fname = fname_size[:i]
        fsize = int(fname_size[i:])
        e = d[fname_size]
        assert e['filename'] == fname
        for region in e['regions']:
            p = region['shape_attributes']
            assert p['name'] == 'point'
            x, y = p['cx'], p['cy']
            ret.append((fname, fsize, x, y))
    return ret

lst = parse_via_json('/Users/gosuke/Downloads/via_project_27Feb2019_22h13m(1).json')

import os

from datetime import datetime

t = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
t

import pandas as pd

DEFAULT_DATA_DIRECTORY = Path('~/repos/lego/data').expanduser()

DEFAULT_DATA_DIRECTORY

DEFAULT_DATA_DIRECTORY = Path('~/repos/lego/data').expanduser()
class LabelDatabase:
    csv_cols = ['file_name', 'file_size', 'x', 'y', 'label']
    def __init__(self, data_directory=DEFAULT_DATA_DIRECTORY):
        data_directory = Path(data_directory)
        try:
            df = pd.read_csv(data_directory/'labels.latest.csv')
            assert set(df.columns) == set(LabelDatabase.csv_cols), df.columns
        except FileNotFoundError:
            df = pd.DataFrame([], columns=LabelDatabase.csv_cols)
        self.df = df
        
    def import_from_vii(self, path):
        cols = LabelDatabase.csv_cols
        f = pd.DataFrame(parse_via_json(path), columns=cols[:4])
        self.df = pd.merge(self.df, f, how='outer', on=cols[:4])

    def save(self):
        ts = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        paths = [
            DEFAULT_DATA_DIRECTORY / 'labels.latest.csv',
            DEFAULT_DATA_DIRECTORY / f'labels.{ts}.csv',
        ]
        for path in paths:
            self.df.to_csv(path, index=False)
            
    def update(self, file_name, file_size, x, y, label):
        row = df.query('file_name==@file_name and file_size==@file_size                         and x=@x and y=@y')
        self.df.loc[row, 'label'] = label
        
    def __len__(self):
        return len(self.df)
    
    def vc(self):
        return pd.DataFrame(dict(
            n=self.df.label.value_counts(dropna=False),
            frac=self.df.label.value_counts(dropna=False, normalize=True),
        )).rename_axis('label', axis=0)
    
    def keys(self, unlabeled_only=True):
        rows = self.df.label.isnull() if unlabeled_only else slice()
        return self.df.loc[rows][['file_name', 'x', 'y']].to_records(index=False).tolist()

db = LabelDatabase()

from functools import lru_cache

class LegoImages:
    def __init__(self):
        self.images = {}
        self.get_crop = lru_cache()(self.get_crop)
        
    def _load(self, fname):
        if fname in self.images: return
        self.images[fname] = PILImage.open(DEFAULT_DATA_DIRECTORY/'raw'/fname)
        
    def get_crop(self, fxy, wh=300, resize=200, grey=False):
        fname, x, y = fxy
        o = wh // 2
        self._load(fname)
        img = self.images[fname].crop((x-o, y-o, x+o, y+o))  # (left, top, right, bottom)
        img = img.resize((resize, resize))
        pixels = img.load()
        x, y = img.size
        x = x // 2
        y = y // 2
        w = 10
        for i in range(x-w, x+w):
            for j in range(y-w, y+w):
                pixels[i, j] = (255, 255, 255)
        pixels[x, y] = (0, 0, 0)
        if grey:
            img = img.convert(mode='L')  # greyscale
        return pilimage2bytes(img)
    
lego_images = LegoImages()

db = LabelDatabase()
db.import_from_vii('/Users/gosuke/Downloads/via_project_27Feb2019_22h13m(1).json')
db.save()

class ImageAndLabel(widgets.VBox):
    def __init__(self):
        self.label = widgets.HTML(
            value='?',
            layout=widgets.Layout(height='100px'),
        )
        self.image = widgets.Image(
            layout=widgets.Layout(height='200px', width='200px'),
        )
        children = (self.label, self.image)
        super().__init__(children)
        
    def update(self, img, label):
        self.image.value = img
        self.label.value = str(label)
        
ial = ImageAndLabel()
ial.layout.width = '200px'
ial

class Landmarks(dict):
    def coordinates(self):
        return list(self.keys())
    def labels(self):
        return list(self.values())

class ImageLabeler:
    def __init__(self, label_name):
        self.clix = 0  # current landmark index
        self.length = len(db)
        self.n_panels = 5
        self.panels = [ImageAndLabel() for _ in range(self.n_panels)]
        self.label_name = label_name
        
        self.landmarks = Landmarks({k: None for k in db.keys()})
        
        self.c = widgets.Label('Click or type on me!')
        self.c.layout.border = '10px solid green'
        self.c.layout.height = '100px'
        tips = widgets.HTML("Some tips")
        tips.layout.height = '100px'
        w = widgets.HBox([*self.panels])
        w = widgets.VBox([w, self.c, tips])
        self.widget = w
        
        self.d = Event(source=self.widget, watched_events=['keydown'])
        self.d.on_dom_event(self.handle_event)                  
        
        self.target_label = 'q'
        
        self.render()
        
    def render(self):
#         clear_output()
        self.update_panels()
        display(self.widget)
    
    def update_panels(self):
        for pix in range(self.n_panels):
            self.update_panel(pix, self.clix + pix - 1)
            
    def update_panel(self, pix, lix):
        if lix < 0:
            lix = self.length + lix
        coord = self.landmarks.coordinates()[lix]
        label = self.landmarks[coord]
        grey = self.label_name == label
        data = lego_images.get_crop(coord, grey=grey)
        caption = [f'Label is {self.label_name}? y/n' if pix == 1 else '']
        yn = 'yes' if label else '?' if label is None else 'no'
        color = 'green' if label else 'grey' if label is None else 'red'
        style = f'font-size: x-large; color: {color};'
        caption += [f'<span style="{style}">{yn}</span>']
        caption += [f'({lix + 1} of {self.length})']
        caption = '<br>'.join(caption)
        caption = f'<div style="text-align: center;">{caption}</span>'
        self.panels[pix].update(data, caption)
        color = 'black' if pix == 1 else 'white'
        self.panels[pix].layout.border = f'5px solid {color}'
        
    def handle_event(self, event):
        key = event['key']
        if key in 'yn':
            self.landmarks[self.landmarks.coordinates()[self.clix]] = key == 'y'
            self.clix += 1
        elif key in ['[', 'ArrowLeft']:
            self.clix -= 1
        elif key in [']', 'ArrowRight']:
            self.clix += 1
        elif key == 'ArrowUp':
            self.padding += 5
        elif key == 'ArrowDown':
            self.padding -= 5
        else:
            self.c.value = f"You pressed: {key}"
            return
        self.clix = max(self.clix, 0)
        self.clix = min(self.length - 1, self.clix)
        self.update_panels()
            
label_name = 'block_1x2'
il = ImageLabeler(label_name)

db.keys()

get_ipython().run_line_magic('debug', '')

from collections import Counter
Counter(il.labels.values())

display(Image('/Users/gosuke/Desktop/blocks.png'))

get_ipython().run_line_magic('pinfo2', 'Event')

hhh.value = 'xya'

hhh

