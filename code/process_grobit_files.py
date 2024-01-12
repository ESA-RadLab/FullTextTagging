import os
from bs4 import BeautifulSoup
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type = str,
        help="Path to the input grobit folder.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the output xml folder.",
    )
    args = parser.parse_args()

    xml_files = [filename for filename in os.listdir(args.input_folder) if filename.endswith(".xml")]
    for xml_file in xml_files:
        file_path = os.path.join(args.input_folder, xml_file)
        with open(file_path, 'r') as tei:
            soup = BeautifulSoup(tei, 'lxml')

        # Find all div tags
        div_tags = soup.find_all('div')
        # Create a new XML document
        new_xml = BeautifulSoup(features="xml")

        # Create a root element for the new XML document
        root = new_xml.new_tag("root")
        new_xml.append(root)
        root.append(soup.title)

        # add abstract to the xml file
        for abstract_p in [p.get_text(separator=' ', strip=True) for p in soup.abstract.find_all('p')]:
            a_tag = new_xml.new_tag("abstract")
            a_tag.string = abstract_p
            root.append(a_tag)
        # add main text
        for div_tag in div_tags:
            paragraphs = [p.get_text(separator=' ', strip=True) for p in div_tag.find_all('p')]
            # Add each paragraph as a new 'p' element under the root
            for paragraph in paragraphs:
                p_tag = new_xml.new_tag("p")
                p_tag.string = paragraph
                root.append(p_tag)

        # Save the new XML document to a file
        # Get the original filename (without extension)
        filename_without_extension = os.path.basename(file_path).rsplit(".")[0]

        # Construct the output XML file path using the original filename
        output_path = os.path.join(args.output_folder, f'{filename_without_extension}.xml')
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(str(new_xml))

def remove_references(tag):
    return tag.name in ['biblstruct', 'listbibl', 'ref']


if __name__ == "__main__":
    main()