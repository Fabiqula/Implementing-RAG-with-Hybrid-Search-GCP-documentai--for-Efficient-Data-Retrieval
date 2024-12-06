import mimetypes
import os
import textwrap
import time
from typing import Any, Dict, List

from google.cloud import documentai_v1 as documentai
from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAIEmbeddings
from pandas import pd
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from tabulate import tabulate
from tiktoken import patch


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "root-array-442413-f5-7994e06766cb.json"

processor_id = '4dac41d96fc1712f'
project_id = 'root-array-442413-f5'
location = 'eu'

file_path = './files/attention_is_all.pdf'

mime_type, _ = mimetypes.guess_type(file_path)
if not mime_type:
    raise ValueError("Unsupported file type. Please provide a valid file.")

with open(file_path, "rb") as file:
    document_content = file.read()

# Setup the processor
opts = {
    "api_endpoint": f"{location}-documentai.googleapis.com"
}
client = documentai.DocumentProcessorServiceClient(client_options=opts)

#  construct the request
name = client.processor_path(project_id, location, processor_id)

raw_document = documentai.RawDocument(content=document_content, mime_type=mime_type)
request = documentai.ProcessRequest(name=name, raw_document=raw_document)

result = client.process_document(request)
document = result.document
print("Document processing complete!")

print(type(document))
