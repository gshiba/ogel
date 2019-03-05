
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

import os

from datetime import datetime

import pandas as pd
import numpy as np

from functools import lru_cache

import base64

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

DEFAULT_DATA_DIRECTORY = Path('~/repos/lego/data').expanduser()

class LabelDatabase:
    idx_cols = ['file_name', 'file_size', 'x', 'y']
    val_cols = ['label']
    all_cols = idx_cols + val_cols
    
    def __init__(self, data_directory=DEFAULT_DATA_DIRECTORY):
        data_directory = Path(data_directory)
        try:
            df = pd.read_csv(data_directory/'labels.latest.csv')
            assert set(df.columns) == set(LabelDatabase.all_cols), df.columns
            df = df.set_index(LabelDatabase.idx_cols)
        except FileNotFoundError:
            df = pd.DataFrame([])
        self.df = df
        
    def import_from_vii(self, path):
        f = pd.DataFrame(parse_via_json(path), columns=LabelDatabase.idx_cols)
        f['label'] = np.nan
        f = f.set_index(LabelDatabase.idx_cols)
        if self.df.empty:
            self.df = f
        else:
            self.df = self.df.merge(f, how='outer', on='label', left_index=True, right_index=True)

    def save(self):
        ts = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        paths = [
            DEFAULT_DATA_DIRECTORY / 'labels.latest.csv',
            DEFAULT_DATA_DIRECTORY / f'labels.{ts}.csv',
        ]
        for path in paths:
            self.df.reset_index().to_csv(path, index=False)
            
    def update(self, key, label):
        self.df.loc[key, 'label'] = label
        
    def __len__(self):
        return len(self.df)
    
    def vc(self):
        return pd.DataFrame(dict(
            n=self.df.label.value_counts(dropna=False),
            frac=self.df.label.value_counts(dropna=False, normalize=True),
        )).rename_axis('label', axis=0)
    
    def keys(self, unlabeled_only=True):
        rows = self.df.label.isnull() if unlabeled_only else slice()
        return self.df.loc[rows].index.to_list()

db = LabelDatabase()
len(db)

db.import_from_vii('/Users/gosuke/Downloads/via_project_27Feb2019_22h13m(1).json')
len(db)

db.save()

class LegoImages:
    def __init__(self):
        self.images = {}
        self.get_crop = lru_cache()(self.get_crop)
        
    def _load(self, fname):
        if fname in self.images: return
        self.images[fname] = PILImage.open(DEFAULT_DATA_DIRECTORY/'raw'/fname)
        
    def get_crop(self, fsxy, wh=300, resize=200, grey=False):
        fname, fsize, x, y = fsxy
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
        self.clear()
        
    def update(self, img, label):
        self.image.value = img
        self.label.value = str(label)

    def clear(self):
        s = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mM8cYLhPwAGgQKRee0MwAAAAABJRU5ErkJggg=='
        self.image.value = base64.b64decode(s)
        self.label.value = ''

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
        
        self.c = widgets.HTML('Click or type on me!')
        button = widgets.Button(description="Save", layout=widgets.Layout(width='auto'))
        button.on_click(self.save)
        w = widgets.HBox([*self.panels])
        w = widgets.VBox([w, self.c, button])
        self.widget = w
        
        self.d = Event(source=self.widget, watched_events=['keydown'])
        self.d.on_dom_event(self.handle_event)                  
        
        self.target_label = 'q'
        
        self.render()
        display(self.widget)
    
    def save(self, _):
        db.save()
        
    def render(self):
        self.update_panels()
        self.c.value = '<pre style="line-height: 12px;">' + db.vc().to_string().replace('\n', '\n') + '</pre>'
    
    def update_panels(self):
        for pix in range(self.n_panels):
            self.update_panel(pix, self.clix + pix)
            
    def update_panel(self, pix, lix):
        if not 0 <= lix < len(self.landmarks):
            self.panels[pix].clear()
            return
        coord = self.landmarks.coordinates()[lix]
        is_target_label = self.landmarks[coord]  # T, F or None
        data = lego_images.get_crop(coord)
        if is_target_label:
            yn, color = 'yes', 'green'
        elif is_target_label is None:
            yn, color = '?', 'grey'
        else:
            yn, color = 'no', 'red'
        ch = 'hjkl;'[pix]
            
        style = f'font-size: x-large; color: {color};'
        style2 = f'font-family: mono; color: grey;'
        caption = [
            f'<span style="{style}">{yn}</span>',
            f'({lix + 1} of {self.length})',
            f'<span style="{style2}">{ch}</span>',
        ]
        caption = '<br>'.join(caption)
        caption = f'<div style="text-align: center;">{caption}</span>'
        
        self.panels[pix].update(data, caption)
        self.panels[pix].layout.border = f'5px solid {color}'
        
    def toggle_label(self, clix):
        if not 0 <= clix < len(self.landmarks):
            return
        coord = self.landmarks.coordinates()[clix]
        current = self.landmarks[coord]
        new = True if current is None else not current
        self.landmarks[coord] = new
        db.update(coord, new)
        
    def falsify_if_unset(self, clix):
        if not 0 <= clix < len(self.landmarks):
            return
        coord = self.landmarks.coordinates()[clix]
        current = self.landmarks[coord]
        new = False if current is None else current
        self.landmarks[coord] = new
        db.update(coord, new)
        
    def handle_event(self, event):
        key = event['key']
        if key in 'hjkl;':
            pix = 'hjkl;'.index(key)
            self.toggle_label(self.clix + pix)
        elif key in ['[', 'ArrowLeft']:
            self.clix -= self.n_panels
        elif key in [']', 'ArrowRight']:
            self.clix += self.n_panels
        elif key in ['Enter']:
            for i in range(self.n_panels):
                self.falsify_if_unset(self.clix + i)
            self.clix += self.n_panels
        elif key == 'ArrowUp':
            self.padding += 5
        elif key == 'ArrowDown':
            self.padding -= 5
        else:
            self.c.value = f"You pressed: {key}"
            return
        self.clix = max(self.clix, 0)
        max_ix = len(self.landmarks) // self.n_panels * self.n_panels
        self.clix = min(max_ix, self.clix)
        self.render()
            
label_name = 'block_1x2'
il = ImageLabeler(label_name)

