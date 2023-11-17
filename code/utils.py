from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import numpy as np


def visualize_boxes(pdf_path, texts, relative=False):
    """
    Visualizes the bounding boxes
    Inputs:
        pdf_path
        texts: List(Text)
        relative: Boolean,
    """
    pages = convert_from_path(pdf_path)

    for page in set([p.page for p in texts]):
        page_texts = [entry for entry in texts if entry.page == page]
        img = pages[page]
        scaler = 200 / 72  # pdf coordinate system
        plt.imshow(img)
        for block in page_texts:
            box = block.bbox
            if relative:
                x, y, w, h = box
            else:
                x0, y0, x1, y1 = box
                w = x1 * scaler - x0 * scaler
                h = y1 * scaler - y0 * scaler
                x = x0 * scaler
                y = y0 * scaler
            xs = [x, x + w, x + w, x, x]
            ys = [y, y, y + h, y + h, y]
            plt.plot(xs, ys)
        plt.show()
