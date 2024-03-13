import sys
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from collections import Counter
import argparse
import constants
from pre_processing import extract_main_text
from custom_types import Paper
from utils import connect_str_to_class, print_confusion_matrix, visualize_boxes

# from utils import visualize_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infile",
        nargs="?",
        const="../data/test_SD_files/Annotation_data.csv",
        default="../data/test_SD_files/Annotation_data.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--mode",
        choices=["test", "run"],
        default="test",
        help="Specify the mode (test or run).",
    )
    parser.add_argument(
        "--input_type",
        choices=["pdf", "xml"],
        default="xml",
        help="Specify the imput type (xml or pdf).",
    )
    parser.add_argument(
        "--max_test_n",
        type=int,
        default=5,
        help="Maximum number of testing data points.",
    )

    # set up the open AI key
    os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

    args = parser.parse_args()

    annotations_filepath = Path(args.infile)

    mode = args.mode

    if mode == "test":
        # Run in test mode
        result_df = process_annotation_file(annotations_filepath, args.max_test_n)
        run_test_mode(
            result_df, poppler_path=constants.poppler_path, input_type=args.input_type
        )
        print("NOT FULLY IMPLEMENTED")
    elif mode == "run":
        # Run in regular mode
        print("NOT IMPLEMENTED")
    else:
        print(f"Error: Unsupported mode - {mode}")
        parser.print_help()


def process_annotation_file(file_path, test_n):
    """
    Read the annotation file and find the ground truth for each paper
    """

    annotations = pd.read_csv(file_path, header=0)
    annotations = annotations.dropna(subset=["PdfRelativePath"])

    # Initialize an empty DataFrame to store the final results
    result_df = pd.DataFrame()

    # Iterate through each unique title in the original DataFrame
    for title in [x for x in annotations["Title"].unique() if str(x) != "nan"][:test_n]:
        # Select rows with the current title
        title_rows = annotations[annotations["Title"] == title]

        # Initialize a dictionary to store the most common answer for each question
        most_common_answers = {}

        # Iterate through each unique question in the original DataFrame
        for question in [
            x for x in annotations["Question"].unique() if str(x) != "nan"
        ]:
            # Select rows with the current title and question
            question_rows = title_rows[title_rows["Question"] == question]
            if len(question_rows) > 0:
                # Count the occurrences of each answer for the current question
                answer_counts = Counter(question_rows["Answer"])
                # Find the most common answer for the current question
                most_common_answer = answer_counts.most_common(1)[0][0]

                # Store the most common answer for the current question in the dictionary
                most_common_answers[question] = most_common_answer

        # Extract additional metadata columns for the current title
        metadata_columns = title_rows.iloc[0][
            ["Url", "Authors", "PdfRelativePath"]
        ]  # Add more columns as needed

        # Combine metadata columns with most common answers dictionary
        row_data = {"Title": title, **metadata_columns.to_dict(), **most_common_answers}

        # Add a new row to the result DataFrame
        result_row = pd.DataFrame([row_data])

        # Append the row to the result DataFrame
        result_df = pd.concat(
            [result_df, result_row], ignore_index=True
        )  # Add a new row to the result DataFrame with the title and most common answers
    return result_df


def run_test_mode(result_df, poppler_path=None, input_type="pdf"):
    """test the performanse with n samples"""
    data_path = "../data/test_SD_files/"
    plot_path = "../plots/"
    llm_embedder: Embeddings = OpenAIEmbeddings(client=None)
    llm: Union[str, BaseLanguageModel] = ChatOpenAI(
        temperature=0.1, model="gpt-3.5-turbo", client=None
    )
    # Lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # List to store misclassified papers
    misclassified_papers = []
    for file in result_df.iterrows():
        # print(file[1])
        file = file[1]
        if input_type == "pdf":
            pdf_path = str(data_path) + str(file["PdfRelativePath"])
            if not os.path.exists(pdf_path):
                print("File not found")
                continue
            paper = Paper(
                file_path=pdf_path,
                llm_embedder=llm_embedder,
                llm=llm,
                input_type=input_type,
            )
            paper.read_paper()
            paper.embed_paper()
        else:
            data_path = "../data/tmp/processed/"
            xml_path = str(
                data_path
                + os.path.basename(file["PdfRelativePath"]).rsplit(".")[0]
                + ".xml"
            )
            print(xml_path)
            if not os.path.exists(xml_path):
                print(xml_path)
                print("File not found")
                continue
            paper = Paper(
                file_path=xml_path,
                llm_embedder=llm_embedder,
                llm=llm,
                input_type=input_type,
            )
            paper.read_paper()
            paper.embed_paper()

        # visualize_boxes(pdf_path = pdf_path, texts=paper.main_text , relative=False)
        query = "methods:, does it use humans or animals as test subjects?"
        llm_answer = paper.query(query, prompt_type="type_0")
        mapping = {
            "Human": "Human",
            "Animal": "Animal",
            "Unsure": "Unsure",
            "Both": "Both",
        }
        mid_answer = connect_str_to_class(llm_answer.answer, class_mapping=mapping)
        if mid_answer == "Non matching output from LLM":
            print(mid_answer)
            print(llm_answer.answer)
        query = (
            "Is the study performed on living test subjects or to cultured cell line?"
        )
        llm_answer = paper.query(query, prompt_type="type_1")
        mapping = {"In-vivo": mid_answer, "In-vitro": "Cells", "Unsure": "Unsure"}
        final_answer = connect_str_to_class(llm_answer.answer, class_mapping=mapping)
        llm_answer.formatted_answer = final_answer

        # query = "Does this article study animals or humans subject or cell samples?"
        # llm_answer = paper.query(query, prompt_type="study_type")
        # mapping = {
        #    "Human": "Human",
        #    "Animal": "Animal",
        #    "In-vitro": "Cells",
        #    "Unsure": "Unsure",
        # }
        new_formatted_answer = connect_str_to_class(
            llm_answer.answer, class_mapping=mapping
        )
        llm_answer.formatted_answer = new_formatted_answer

        # Append true label and predicted label for confusion matrix
        # print(file)
        # print(type(file))
        # print(file.keys())
        true_labels.append(file["This article concerns "])
        predicted_labels.append(llm_answer.formatted_answer)

        if (file["This article concerns "] != llm_answer.formatted_answer) and (
            llm_answer.formatted_answer != "Unsure"
        ):
            print("\n")
            print(file["This article concerns "])
            print(file["Please give details"])
            print("Prediction:")
            print(mid_answer)
            print(final_answer)
            print(llm_answer.answer)
            print(llm_answer.formatted_answer)
            print(pdf_path)
            # visualize_boxes(
            #    pdf_path=pdf_path,
            #    texts=paper.main_text,
            #    relative=False,
            #    save_path=plot_path,
            #    poppler_path=poppler_path,
            # )
            print("CONTEXT:")
            print(llm_answer.context)
            misclassified_papers.append(misclassified_papers.append(paper))

    print("Labels:")
    print(true_labels)
    print(predicted_labels)
    print_confusion_matrix(true_labels, predicted_labels)


if __name__ == "__main__":
    # sys.exit(main(sys.argv[1:]))
    main()
