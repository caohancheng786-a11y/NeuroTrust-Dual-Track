# -*- coding: utf-8 -*-
from PIL import Image
SRC = r'F:\研究生gpt\周老师\MDPI准备投稿\05_高清图Figures\Figure1_SEM_core_model_EN_600dpi.png'
OUT = r'C:\Users\ASUS\Doubao\chats\2026-09-04\new-chat-1\fig1_preview.png'
im = Image.open(SRC)
im.thumbnail((1900, 1900))
im.save(OUT)
print(im.size)
