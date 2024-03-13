from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from help_types import Text
from bs4 import BeautifulSoup

# from difflib import SequenceMatcher
import numpy as np


def visualize_boxes(pdf_path, texts, relative=False, save_path=None, poppler_path=None):
    """
    Visualizes the bounding boxes.
    Inputs:
        pdf_path
        texts: List(Text)
        relative: Boolean,
    """
    pages = convert_from_path(pdf_path, poppler_path=poppler_path)

    for page in set([p.page for p in texts]):
        page_texts = [entry for entry in texts if entry.page == page]
        img = pages[page]
        scaler = 200 / 72  # pdf coordinate system

        # Create a new figure and axis for each page
        fig, ax = plt.subplots()

        ax.imshow(img)

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
            ax.plot(xs, ys)

        if save_path:
            # Extract the filename from the pdf_path
            filename = os.path.splitext(os.path.basename(pdf_path))[0]
            plot_filename = f"{filename}_page{page}.png"
            plot_path = os.path.join(save_path, plot_filename)

            # Save the plot
            canvas = FigureCanvas(fig)
            canvas.print_png(plot_path)
            print(f"Plot saved at: {plot_path}")
        else:
            # Show the plot if save_path is not provided
            plt.show()

        # Clear the axis for the next iteration
        ax.clear()


def visualize_boxes_old(
    pdf_path, texts, relative=False, save_path=None, poppler_path=None
):
    """
    Visualizes the bounding boxes
    Inputs:
        pdf_path
        texts: List(Text)
        relative: Boolean,
    """
    pages = convert_from_path(pdf_path, poppler_path=poppler_path)
    page = None
    for page in set([p.page for p in texts])[:15]:
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

        if save_path:
            # Extract the filename from the pdf_path
            filename = os.path.splitext(os.path.basename(pdf_path))[0]
            plot_filename = f"{filename}_page{page}.png"
            plot_path = os.path.join(save_path, plot_filename)

            # Save the plot
            plt.savefig(plot_path, format="png")
            print(f"Plot saved at: {plot_path}")
        else:
            # Show the plot if save_path is not provided
            plt.show()


def connect_str_to_class(output_string: str, class_mapping: dict) -> str:
    """
    Connects output string from the llm to the corresponding class in the dictionary.
    output_string: str
    class_mapping: Dict[str, Type[BaseLanguageModel]]
    """

    for class_name, class_instance in class_mapping.items():
        if class_name.lower() in output_string.lower():
            return class_instance

    return "Non matching output from LLM"


def print_confusion_matrix(true_labels, predicted_labels):
    """
    Print the confusion matrix.
    true_labels: List of true labels
    predicted_labels: List of predicted labels
    """
    confusion_df = pd.DataFrame(
        confusion_matrix(true_labels, predicted_labels),
        index=sorted(set(true_labels + predicted_labels)),
        columns=sorted(set(true_labels + predicted_labels)),
    )

    print("Confusion Matrix:")
    print(confusion_df)


def read_xml_file(file_path):
    """
    Read xml file, checks if the embeddings are also saved
    true_labels: List of true labels
    return: List of Text objects with or without embeddings
    """
    main_text = []
    with open(file_path, "r", encoding="utf8") as f:
        soup = BeautifulSoup(f, features="html.parser")  # html.parser
    print("tags")
    print([tag.name for tag in soup.find_all()])
    paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
    if soup.find("embeddings"):
        embeddings = soup.embeddings.find_all("e")
        if len(embeddings) == len(paragraphs):
            print("Embeddings were found in xml")
            for embed, parag in zip(embeddings, paragraphs):
                main_t = Text(
                    text=parag,
                    page=0,
                    embeddings=[float(e) for e in embed.contents[0][1:-1].split(", ")],
                )
                main_text.append(main_t)
        else:
            print("not all embeddings were found")
            print(len(embeddings))
            print(len(paragraphs))
            for paragraph in paragraphs:
                main_t = Text(text=paragraph, page=0)
                main_text.append(main_t)
    else:
        for paragraph in paragraphs:
            main_t = Text(text=paragraph, page=0)
            main_text.append(main_t)
    return main_text
