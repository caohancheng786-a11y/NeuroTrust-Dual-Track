# -*- coding: utf-8 -*-
from docx import Document
SRC = r'F:\研究生gpt\周老师\MDPI准备投稿\01_正文Manuscript\Manuscript.docx'
d = Document(SRC)
for i, t in enumerate(d.tables):
    head = " | ".join(c.text for c in t.rows[0].cells)
    print("===== TABLE %d (rows=%d) header: %s" % (i, len(t.rows), head[:120]))
    if ('NT' in head and 'ST' in head) or 'path' in head.lower() or 'β' in head:
        for r in t.rows:
            print("   " + " | ".join(c.text for c in r.cells))
