from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import time
import asyncio
import os
import pinecone
import glob
import csv
import tiktoken


load_dotenv()
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
index_name = os.getenv("PINECONE_INDEX_NAME")
embeddings = OpenAIEmbeddings()

tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def get_pdf_text(filePath: str):
    text = ""
    with open(filePath, 'rb') as pdf:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks


def main(path: str):

    with open('embedding_text.csv', mode='a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['source', 'embedding'])
        pdf_files = glob.glob(path + '/**/*.pdf', recursive=True)
        count = 0
        for pdf in pdf_files:
            try:
                pdf = "Industrial Relations Act 1967 (Act 177).PDF"
                file_name = os.path.basename(pdf)
                text = get_pdf_text(pdf)
                chunks = get_text_chunks(text)
                for chunk in chunks:
                    writer.writerow([file_name, chunk])
                print(count)
                count = count + 1
            except:
                print(file_name)

    # await asyncio.gather(task1, task2)
    # coro = asyncio.create_task(task1)
    # print("Main is doing other things")
    # await coro
    # tasks = [task1, task2]
    # await asyncio.gather(*tasks)
main("./pdfs")
