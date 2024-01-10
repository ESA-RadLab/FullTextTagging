# the code adapts to https://towardsdatascience.com/extracting-text-from-pdf-files-with-python-a-comprehensive-guide-9fc4003d517
# To read the PDF
import PyPDF2

# To analyze the PDF layout and extract text
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
from help_types import Text

# fine tune the miner parameters
from pdfminer.layout import LAParams

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
    line_lengths = []
    for text_line in element:
        char_formats = []
        if isinstance(text_line, LTTextContainer):
            line_length = 0
            # Iterating through each character in the line of text
            if len(text_line) > 5:
                for character in text_line:
                    if isinstance(character, LTChar):
                        # Append the font name of the character and the size
                        char_formats.append((character.fontname, round(character.size)))
                        # char_formats.append(("dummy", round(character.size)))
                        line_length += 1
        if char_formats:
            # find the most common format in line
            line_formats.append(max(set(char_formats), key=char_formats.count))
            line_lengths.append(line_length)

    # Return a tuple with the text in each line along with its format
    return (line_text, line_formats, line_lengths)


def filter_other_fonts(text_per_page):
    """
    This function removes all lines that are not written with the most common font type.
    Inputs: text_per_page: dictionary for each pages linewise text, and format
    Outputs: Only the main text as a one string
    """
    main_text = []
    flat_list = []
    # find the most common font

    # Flatten the list of pages, chapters, and fonts

    flattened_fonts = [
        font
        for page_key, page in text_per_page.items()
        for chapter_fonts in page[1]
        for font in chapter_fonts
    ]

    flattened_lengths = [
        font
        for page_key, page in text_per_page.items()
        for chapter_fonts in page[-1]
        for font in chapter_fonts
    ]
    # Create a list of (font, character count) tuples for each font
    font_character_tuples = list(zip(flattened_fonts, flattened_lengths))

    # Use Counter to count occurrences of each font
    font_counts = Counter(font_character_tuples)

    # Find the most common font for the entire document
    most_common_format = font_counts.most_common(1)[0][0][0]
    # for page in text_per_page:
    # for block, length in zip(text_per_page[page][1], text_per_page[page][4]):
    #    #for item, l in zip(block, length):
    #    #    flat_list.append(item)
    #    #print(flat_list)
    # test = Counter(chain(flat_list))
    # most_common_format = test.most_common(1)[0][0]
    # filter based on the most common font
    for page in text_per_page:
        for format, text, bbox in zip(
            text_per_page[page][1], text_per_page[page][0], text_per_page[page][3]
        ):
            if most_common_format in format:
                main_t = Text(text=text, page=int(page[-1]), bbox=bbox)
                main_text.append(main_t)
    return main_text


def extract_main_text(pdf_path):
    """
    pdf_path: relative or absolute path of the pdf in question
    output: dictionary with page_x keys and values of the pdf content
    """
    laparams = LAParams(
        line_margin=1,
        char_margin=5,
        boxes_flow=-0.6,
        word_margin=0.2,
        all_texts=False,
        detect_vertical=False,
    )

    text_per_page = {}
    all_formats = []
    for pagenum, page in enumerate(
        extract_pages(pdf_path, maxpages=30, laparams=laparams)
    ):
        if pagenum == 29:
            print("Maximum number of pages (30) reached when reading the pdf")
        # Initialize the variables needed for the text extraction from the page
        page_text = []
        line_format = []
        page_content = []
        bounding_boxs = []
        line_lengths = []
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
                (line_text, format_per_line, line_length) = text_extraction(element)
                if len(line_text) > 5:
                    # Append the text of each line to the page text
                    page_text.append(line_text)
                    # Append the format for each line containing text
                    line_format.append(format_per_line)
                    page_content.append(line_text)
                    line_lengths.append(line_length)
                    # bounding_boxs.append(element.bbox)
                    # get the bottom of the page, as the coordinate system is fixed at the left bottom corner
                    bottom = page.bbox[3]
                    x0 = element.x0
                    y1 = bottom - element.y0
                    x1 = element.x1
                    y0 = bottom - element.y1
                    # x0, y0_orig, x1, y1_orig = element.bbox
                    # y0 = page.mediabox[3] - y1_orig
                    # y1 = page.mediabox[3] - y0_orig
                    bounding_boxs.append([x0, y0, x1, y1])

        # Create the key of the dictionary
        dctkey = "Page_" + str(pagenum)
        all_formats.append(line_format)
        # Add the list of list as the value of the page key
        text_per_page[dctkey] = [
            page_text,
            line_format,
            page_content,
            bounding_boxs,
            line_lengths,
        ]

    # Only include taxt in the most common font
    main_text = filter_other_fonts(text_per_page)

    return main_text
    # Closing the pdf file object
    # pdfFileObj.close()
