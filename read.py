# package imports
import pandas as pd
import numpy as np
import fitz
import re
import os
import openai
import pytesseract
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def clean_ocr_using_llm(raw_text):
    """
    Clean and chunk OCR text using GPT.

    Input:
        raw_text (str): Raw text extracted from OCR of a PDF document
    
    Output:
        cleaned_text (str): Cleaned and chunked text
    """
    # Define system and user messages for GPT
    system_msg = 'You are a helpful assistant who helps clean and chunk data derived from OCR on a PDF Document'
    user_msg = f"""Given the text from the OCR extract of a PDF document, do the following:

                Cleaning:
                - Remove any line numbers or unnecessary recurring symbols in each line.
                - Correct any obvious OCR errors such as typos or missing characters or line spaces.
                - Remove any unnecessary punctuation or symbols that do not add value to the text.
                - Ensure that the text flows naturally and is easy to read.
                - Do **not** rephrase; use the original text.

                Chunking:
                The text will be given to you in overlapping chunks. Do the following:

                - Each chunk is the largest set of words containing complete information about one or more contexts.
                - Try to make the chunks as large as possible.
                - Add 3 extra line spaces at the end of each chunk, other than this make sure there are no more than 1 line spaces in the text.
                - Do **not** include any other text.
                - **Only** return the original cleaned, chunked text.


                **Use original text, Do Not Rephrase or add any new text and only return the cleaned, chunked text**

                Raw OCR Text: {raw_text}

                """

    # Use GPT-4o model to process the text
    response = openai.chat.completions.create(model="gpt-4o",
                                            messages=[{"role": "system", "content": system_msg},
                                            {"role": "user", "content": user_msg}])
    
    return response.choices[0].message.content


def read(files, method='pytesseract'):
    """
    Extract text from PDF files using OCR methods.

    Input:
        files (list): List of PDF file paths
        method (str): OCR method ('pytesseract' or 'easyocr'), default is 'pytesseract'

    Output:
        text_dict (dict): Dictionary with PDF file names as keys and extracted text as values
    """
    text_dict = {}

    for file in files:
        text_dict[file] = {}

        # Process each page in the PDF document
        with fitz.open(file) as doc:
            number_pages = 0
            for page in doc:
                # Extract text from the PDF page
                pix = page.get_pixmap(dpi=200)

                if method == 'pytesseract':
                    # Use Pytesseract for OCR
                    text = pytesseract.image_to_string(
                        pix_to_image(pix), config=r"--psm 6"
                    )
                else:
                    # Use EasyOCR for OCR
                    easyocr_output_dict = reader.readtext(pix_to_image(pix), paragraph=False)
                    for source, text_extract, confidence in easyocr_output_dict:
                        text = "\n".join([text_extract])

                # Split text into lines
                lines_page = text.split("\n")
                text_dict[file][number_pages] = lines_page
                number_pages += 1

                # break for testing
                # if number_pages == 10:
                #     break

        # Write extracted text to a text file
        with open(f"{file.split('.')[0]}.txt", "w") as f:
            for page_number, lines in text_dict[file].items():
                for line in lines:
                    f.write(line + "\n")

    return text_dict


def pix_to_image(pix):
    """
    Convert Fitz pixmap to image.

    Input:
        pix (fitz.Pixmap): Fitz Pixmap object
    
    Output:
        img (np.ndarray): Image as numpy array
    """
    bytes = np.frombuffer(pix.samples, dtype=np.uint8)
    img = bytes.reshape(pix.height, pix.width, pix.n)
    return img