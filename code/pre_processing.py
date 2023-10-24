# the code adapts to https://towardsdatascience.com/extracting-text-from-pdf-files-with-python-a-comprehensive-guide-9fc4003d517
# To read the PDF
import PyPDF2

# To analyze the PDF layout and extract text
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure

# fine tune the miner parameters
from pdfminer.layout import LAParams

# To extract text from tables in PDF
import pdfplumber

from collections import Counter
from itertools import chain

## help functions for data preprocessing


# Create a function to extract text
def text_extraction(element):
    """
    element: page object by the pdfminer
    output: tuple with the element text and line wise formats
    """
    # Extracting the text from the in-line text element
    line_text = element.get_text()

    # Find the formats of the text
    line_formats = []
    for text_line in element:
        char_formats = []
        if isinstance(text_line, LTTextContainer):
            # Iterating through each character in the line of text
            for character in text_line:
                if isinstance(character, LTChar):
                    # Append the font name of the character and the size
                    char_formats.append((character.fontname, round(character.size)))
        if char_formats:
            # find the most common format in line
            line_formats.append(max(set(char_formats), key=char_formats.count))

    # Return a tuple with the text in each line along with its format
    return (line_text, line_formats)


def filter_other_fonts(text_per_page):
    """
    This function removes all lines that are not written with the most common font type.
    Inputs: text_per_page: dictionary for each pages linewise text, and format
    Outputs: Only the main text as a one string
    """
    main_text = ""
    i = 0
    flat_list = []
    for page in text_per_page:
        for block in text_per_page[page][1]:
            for item in block:
                flat_list.append(item)
    test = Counter(chain(flat_list))
    most_common_format = test.most_common(1)[0][0]

    for page in text_per_page:
        for format, text in zip(text_per_page[page][1], text_per_page[page][0]):
            if most_common_format in format:
                main_t = text
                main_text += main_t
    return main_text


def extract_main_text(pdf_path):
    """
    pdf_path: relative or absolute path of the pdf in question
    output: dictionary with page_x kays and values of the pdf content
    """
    laparams = LAParams(
        line_margin=0.6, char_margin=3, boxes_flow=-0.6, word_margin=0.2
    )
    text_per_page = {}
    all_formats = []
    for pagenum, page in enumerate(extract_pages(pdf_path, laparams=laparams)):
        # Initialize the variables needed for the text extraction from the page
        page_text = []
        line_format = []
        page_content = []
        # Open the pdf file
        # pdf = pdfplumber.open(pdf_path)
        # Find all the elements
        page_elements = [(element.y1, element) for element in page._objs]
        # Sort all the elements as they appear in the page
        # page_elements.sort(key=lambda a: a[0], reverse=True)

        # Find the elements that composed a page
        for component in page_elements:
            # Extract the element of the page layout
            element = component[1]

            # Check if the element is a text element
            if isinstance(element, LTTextContainer):
                # Use the function to extract the text and format for each text element
                (line_text, format_per_line) = text_extraction(element)
                # Append the text of each line to the page text
                page_text.append(line_text)
                # Append the format for each line containing text
                line_format.append(format_per_line)
                page_content.append(line_text)

        # Create the key of the dictionary
        dctkey = "Page_" + str(pagenum)
        all_formats.append(line_format)
        # Add the list of list as the value of the page key
        text_per_page[dctkey] = [
            page_text,
            line_format,
            page_content,
        ]

    # Only include taxt in the most common font
    main_text = filter_other_fonts(text_per_page)

    return main_text
    # Closing the pdf file object
    # pdfFileObj.close()
