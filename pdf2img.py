import numpy as np
import cv2
from pdf2image import convert_from_path
from sys import argv
import os

filename = argv[1][:-4]
pages = convert_from_path(f'PDF/{argv[1]}', 600)
os.mkdir(f'Sheets/{filename}')
for i, page in enumerate(pages):
    page.save(f'Sheets/{filename}/sheet_{i}.png', 'PNG')
