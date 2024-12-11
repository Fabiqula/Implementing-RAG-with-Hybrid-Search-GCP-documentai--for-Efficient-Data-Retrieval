"""
Main application for processing pdf documents with Google Document AI, generating hybrid embeddings,
and storing them in Pinecone for RAG-based retrieval.
File types accepted: PDF, GIF, TIFF, JPEG, PNG, BMP, WEBP.

Steps:
1. Process documents using Google Cloud Document AI.
2. Extract and process text data from the document's layout.
3. Generate embeddings using OpenAI's embedding model by using langchain ability to change dimensions of embeddings.
4. Set up Pinecone for efficient vector-based search storage.
5. Train and generate sparse embeddings using BM25.
6. Setup vectors in correct format required by database.
7. Upsert generated vectors.
8. Embedd the query and create a hybrid slider function to decide how much sparse vectors should be used.
9. Create logic for asking questions.
10. Display answers from Pinecone in tabulate form.

Environment variables and configuration:
- GOOGLE_APPLICATION_CREDENTIALS must point to your service account JSON key.
- Pinecone API_KEY in .env file in root for serverless index AWS cloud.
- OPENAI API_KEY in .env file in root for Embedding Client
"""
import mimetypes
import os
import pickle
import textwrap
import time
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAIEmbeddings
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from tabulate import tabulate
import tiktoken

# Path to a file for OCR
file_path = './files/Churchill_Beaches_Speech.pdf'

# Global variables:
    # Wrapp and truncate width and length:
MAX_WIDTH=45
MAX_CHAR_LENGTH=200

    # API Credentials
load_dotenv(find_dotenv(), override=True)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Client_options.json"

# Configure embedding client.
EMBEDDINGS_DIMENSIONS = 512
embedding_client = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=EMBEDDINGS_DIMENSIONS)

# Create an index_name for Pinecone Database.
# Pinecone allows for one index only in their free plan, so each time new document will be indexed old will be deleted.
index_name = "hybridsearch"

# External API function moved to global so it could be mocked by unittest.patch(line 270: def test_create_embeddings())
def tokens_from_strings(string, encoding_name):
    """
        Calculate the number of tokens in a string using the specified encoding.
        Function moved to global scope for unittest access.

        Args:
            string (str): Input text string.
            encoding_name (str): Encoding type to determine token count.

        Returns:
            int: Number of tokens calculated.
        """
    enc = tiktoken.get_encoding(encoding_name)
    tokens = len(enc.encode(string))
    return tokens

def my_main():
    """
    Main function to process a document, extract data, and perform embeddings.
    Handles document loading, processing, and data extraction from OCR pipelines.
    """

    # GOOGLE documentai info and path to file.
    # You must create a processor before running this code.
    processor_id = '4dac41d96fc1712f'
    project_id = 'root-array-442413-f5'
    location = 'eu'

    mime_type, _ = mimetypes.guess_type(file_path)

    def process_document(
        loc_project_id: str,
        loc_location: str,
        loc_processor_id: str,
        loc_processor_version: str,
        loc_file_path: str,
        loc_mime_type: str,
        process_options: Optional[documentai.ProcessOptions] = None,
    ) -> documentai.Document:
        """
        Process a document using the Google Document AI OCR service.

        Args:
            loc_project_id (str): Google Cloud project ID.
            loc_location (str): The location of the Google Cloud region.
            loc_processor_id (str): The ID of the Document AI processor to use.
            loc_processor_version (str): The version of the processor.
            loc_file_path (str): Path to the document file to process.
            loc_mime_type (str): MIME type of the document.
            process_options (Optional[documentai.ProcessOptions]): Options for processing the document.

        Returns:
            documentai.Document: Parsed document response with extracted data.
                """
        _ = process_options
        # You must set the `api_endpoint` if you use a location other than "us".
        client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(
                api_endpoint=f"{loc_location}-documentai.googleapis.com"
            )
        )

        # You must create a processor before running this code.
        name = client.processor_version_path(
            loc_project_id, loc_location, loc_processor_id, loc_processor_version
        )
        # Read the file into memory
        with open(loc_file_path, "rb") as image:
            image_content = image.read()

        # Configure the process request
        request = documentai.ProcessRequest(
            name=name,
            raw_document=documentai.RawDocument(content=image_content, mime_type=loc_mime_type),
            # Only supported for Document OCR processor
            process_options=None,
        )

        loc_result = client.process_document(request=request)
        print("Document processing complete!")

        # For a full list of `Document` object attributes, reference this page:
        # https://cloud.google.com/document-ai/docs/reference/rest/v1/Document
        return loc_result.document


    # Create dictionary with paragraphs as strings with its location for each page.
    def create_dict(loc_document):
        """
        Extract paragraphs from the document, organizing their text and metadata.

        Args:
            loc_document (documentai.Document): Parsed document from OCR.

        Returns:
            List[Dict]: List of paragraphs with extracted text and metadata.
        """
        document_text = loc_document.text
        list_block = []
        for page_num, page in enumerate(loc_document.pages, start=1):
            for block in page.blocks:
                text = ''
                # Iterate through each text segment in the block's text_anchor
                for segment in block.layout.text_anchor.text_segments:
                    # Use the segment's start and end index to extract the text
                    start_idx = segment.start_index
                    end_idx = segment.end_index
                    text += document_text[start_idx:end_idx]

                # Extracting the bounding polygon (location of the block on the page)
                location = [
                    {
                        'x': vertex.x,
                        'y': vertex.y
                    }
                    for vertex in block.layout.bounding_poly.normalized_vertices
                ]
                block_dict = {
                    'paragraph': text,
                    'metadata': {
                        'page': page_num,
                        'location': location
                    }
                }

                # block_dict = {
                #     'page': ,
                #     'paragraph': text,
                #     'location': location
                # }
                list_block.append(block_dict)
        return list_block


    # Format 'location' column in a df
    def format_location(location):
        """
        Format the location data for better readability.

        Args:
            location (Any): List of location dictionaries.

        Returns:
            str: Formatted location string.
        """
        # If the location is a list of dictionaries, convert each dict to a string
        if isinstance(location, list):
            return '\n'.join([f"x:{loc['x']}, y:{loc['y']}" for loc in location])
        return str(location)


    def tabulate_df_copy(df):
        """
        Wrap text in DataFrame cells for better visualization using tabulate.

        Args:
            df (pd.DataFrame): DataFrame to visualize.

        Returns:
            str: Tabulated DataFrame with wrapped text.
        """
        width = 70
        for col in df.columns:
            df.loc[:,col] = df.loc[:,col].apply(lambda x: '\n'.join(textwrap.wrap(str(x), width)) if isinstance(x, str) else x)
        result = tabulate(df, headers='keys', tablefmt='simple')
        return result


    # def tokens_from_strings(string, encoding_name):
    #     enc = tiktoken.get_encoding(encoding_name)
    #     tokens = len(enc.encode(string))
    #     return tokens


    # Create batch embedding function.
    def create_embeddings(
            df: pd.DataFrame,
            num_rows: int = 2,
            encoding_name: str = 'cl100k_base',
            max_tokens: int = 8191,
            price_per_token: float = 0.13 / 1000000
    ) -> List[Dict]:
        """
        Process a DataFrame to create embeddings for specified rows, batching by token limit.

        Args:
            df (pd.DataFrame): DataFrame containing data to process.
            num_rows (int): Number of rows to process from the DataFrame.
            encoding_name (str): Encoding name for token calculation.
            max_tokens (int): Maximum tokens allowed per batch.
            price_per_token (float): Cost per token for API usage.

        Returns:
            List[Dict]: List of dictionaries containing embeddings, page, and location details in a single list.
        """

        result_list = []
        batch_page = []
        batch_location = []
        current_batch_text = []
        current_tokens = 0
        total_batches = 0
        num_batch = 0
        total_tokens = 0

        start_time = time.time()

        for idx, row in df.head(num_rows).iterrows():
            text = row['paragraph']

            # set the condition for filtering only strings and not empty strings.
            if not isinstance(text, str) or not text.strip():
                continue
            # Calculate the number of tokens from current iteration text.
            tokens = tokens_from_strings(text, encoding_name)

            # Prepare the batch with max_token limit to be embedded.
            if current_tokens + tokens >= max_tokens:
                try:
                    embeddings = embedding_client.embed_documents(current_batch_text)
                    for i, text in enumerate(current_batch_text):
                        result_list.append(
                            {'page': batch_page[i],
                             'paragraph': current_batch_text[i],
                             'location': batch_location[i],
                             'embeddings': embeddings[i],
                             })

                except Exception as e:
                    print(f"Error while creating embeddings for batch: {e}")
                    continue  # Skip this batch and move to the next

                # Reset batch and start a new one with the current text.
                current_batch_text = [text]
                batch_page = [row['page']]
                batch_location = [row['location']]
                current_tokens = tokens
                total_batches += 1


            else:
                current_batch_text.append(text)
                batch_page.append(row['page'])
                batch_location.append(row['location'])
                current_tokens += tokens

            total_tokens += tokens
            # print(f"Current batch updated: {current_batch_text}") - Debugging line.
            # print('-'*50)
        # if current_text_batch won't reach max_token limit (because last batch or short text) embedd all of it in one batch.
        if current_batch_text:
            embeddings = embedding_client.embed_documents(current_batch_text)
            # print([embd[0:3]for embd in embeddings]) -  Debugging line.

            for i, text in enumerate(current_batch_text):
                result_list.append(
                    {'page': batch_page[i],
                     'paragraph': current_batch_text[i],
                     'location': batch_location[i],
                     'embeddings': embeddings[i],
                     })
            total_batches += 1
            # print(f"Result list :{[{'embeddings': res['embeddings'][:3]} for res in result_list]}") -  Debugging line.

        end_time = time.time()
        duration = end_time - start_time

        # Print the statistics
        print("\nEmbedding stats:")
        print(f'Duration: {duration:.2f} seconds')
        print(f'Number of batches: {total_batches:,}')
        print(f'Total number of tokens: {total_tokens:,}')
        print(f'Costs: ${total_tokens * price_per_token:.6f}\n')

        return result_list


    # Test function for emb
    # Mock function for tokens_from_strings
    def mock_tokens_from_strings(text, encoding_name):
        return len(text.split())  # Example: number of words as token count


    # Test function for create_embeddings(
    #         df: pd.DataFrame,
    #         num_rows: int = 2,
    #         encoding_name: str = 'cl100k_base',
    #         max_tokens: int = 8191,
    #         price_per_token: float = 0.13 / 1000000
    # ) -> List[Dict]:

    # Create mocking function for calculating tokens
    def test_create_embeddings():
        """
        Test function for create_embeddings using mocks for token counting and embeddings.
        Simulates expected embeddings.
        """
        # Create sample input DataFrame
        input_df = pd.DataFrame({
            "page": [1, 2],
            "location": ["page1", "page2"],
            "paragraph": ["this is test", "test"]
        })

        # Dummy embeddings
        dummy_embedding = [{"token": i} for i in range(4)]  # Example embedding for a single text

        # Mock both external dependencies
        with patch('__main__.tokens_from_strings', side_effect=mock_tokens_from_strings):
            with patch('langchain_openai.OpenAIEmbeddings.embed_documents', return_value=[dummy_embedding] * 2):
                # Call the function
                output = create_embeddings(input_df, max_tokens=10)

        # Define expected output
        expected_output = [
            {'page': 1, 'paragraph': "this is test", 'location': "page1", 'embeddings': dummy_embedding},
            {'page': 2, 'paragraph': "test", 'location': "page2", 'embeddings': dummy_embedding}
        ]

        # Manual comparison
        if output == expected_output:
            print("Test result: OK (the outputs match expected results)")
        else:
            print("Test failed: Output does not match expected result")
            print("Expected:", expected_output)
            print("Got:", output)


    # Wrapp and truncate the text before tabulate.
    def wrap_truncate(text, width=MAX_WIDTH, length=MAX_CHAR_LENGTH):
        if not isinstance(text, str):
            text=str(text)
            if len(text) >= length:
                text = text[:length] + '...'
        else:
            text=text
        return '\n'.join(textwrap.wrap(text, width))


    def create_pinecone_index(index_name):
        """
        Create or reset a Pinecone index for vector similarity search.

        Args:
            index_name (str): Name of the index to create.

        Returns:
            Pinecone.Index: The Pinecone index ready for use.
        """
        pc = Pinecone()

        start_time = time.time()

        # Checking if this index already exist
        existing_indexes = pc.list_indexes().names()

        if existing_indexes:
            for idx in existing_indexes:
                pc.delete_index(idx)
            print("Deleted old index.")

        # For sparse-dense index in Pinecone, use dot product metric.
        pc.create_index(
            name=index_name,
            dimension=512,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Created new index: {index_name}")
        end_time = time.time()
        duration = end_time - start_time

        print(f"Duration for the Pinecone: {duration:.2f} seconds")
        # Wait for index to become ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

        pinecone_index = pc.Index(index_name)
        return pinecone_index


    # train sparse encoder on a corpus
    def sparse_encoder(train_df):
        """
        Trains a BM25 sparse encoder on a given corpus of paragraphs.

        Args:
            train_df (pd.DataFrame): DataFrame containing a 'paragraph' column with text data.

        Returns:
            BM25Encoder: A trained sparse BM25 encoder for encoding documents.
        """
        bm25 = BM25Encoder()
        enc = bm25.fit(train_df.get('paragraph').astype(str).to_list())
        return enc

    # Reshape our data to match vector requirements for Pinecone index
    def create_vector_for_uploading(blocks, embedding_model):
        """
        Transforms processed text blocks into dense and sparse vectors for Pinecone upload.

        Args:
            blocks (list): List of data blocks containing embeddings, locations, and paragraph text.
            embedding_model: Model used for encoding embeddings.

        Returns:
            list: A list of vectors (dense and sparse) ready for Pinecone storage, with metadata.
        """
        dense_and_sparse_vectors = []
        for idx, block in (enumerate(blocks)):

            embeddings = block['embeddings']
            page = block['page']
            location = [(f"{str(k)}{d[0]}: {str(v)}") for d in enumerate(block['location']) for k, v in d[1].items()]
            paragraph = block['paragraph']

            sparse_vectors = embedding_model.encode_documents(paragraph)
            if sparse_vectors['indices'] and sparse_vectors['values']:
                # Proceed with upserting the vector if sparse vector is not empty
                dense_and_sparse_vectors.append(
                    {
                        'id': f"page_paragraph_no: {page}_{idx}",
                        'values': embeddings,
                        'sparse_values': sparse_vectors,
                        'metadata': {
                            'page': page,
                            'location': location,
                            'paragraph': paragraph
                        }
                    })
        return dense_and_sparse_vectors


    # Prepare a table our reshaped vector for tabulate
    def vector_preview(reshaped_vector, char_len=400, int_len=40):
        """
            Prepares and formats reshaped vectors into a human-readable table for visualization.

            Args:
                reshaped_vector (list): List of reshaped vector data to visualize.
                char_len (int, optional): Maximum number of characters per text snippet. Defaults to 400.
                int_len (int, optional): Maximum number of characters for embedding previews. Defaults to 40.

            Returns:
                str: A formatted table string for visualization using tabulate.
            """
        table = []
        for block in reshaped_vector[:4]:
            table.append([
                str(block['metadata']['page'])[:int_len],
                block['metadata']['paragraph'][:char_len],
                [element[:10] for element in block['metadata']['location']],
                str(block['values'])[:int_len],
                str(block['sparse_values']['values'])[:int_len]
            ])

        headers = ['page', 'paragraph', 'location', 'dense_embeddings', 'sparse_values']
        colalign = ['left', 'left', 'left', 'left', 'left']

        table = [['\n'.join(textwrap.wrap(str(item), width=34)) for item in element] for element in table]

        tabulate_vector = (tabulate(table, headers=headers, tablefmt='simple', colalign=colalign))
        return tabulate_vector


    def convert_string_query_to_vectors(query: str) -> Dict[str, Any]:
        """
        Converts a string query into both dense and sparse embeddings for vector search.

        Args:
            query (str): Input string query to convert into embeddings.

        Returns:
            dict: Dictionary containing dense and sparse vector representations of the query.
        """
        dense_vector = embedding_client.embed_query(query)
        sparse_vector = enc.encode_queries(query)

        return {
            "dense": dense_vector,
            "sparse": sparse_vector
        }

    def hybrid_search(dense, sparse, factor: float):
        """Hybrid vector scaling using a convex combination

        h * dense + (1 - h) * sparse

        Args:
            dense: Array of floats representing the dense embedding
            sparse: a dict of `indices` and `values` representing the sparse embedding
            factor: float between 0 and 1 where 0 == sparse only
                   and 1 == dense only
        """
        if h < 0 or h > 1:
            raise ValueError("h must be between 0 and 1")
        # scale sparse and dense vectors to create hybrid search vecs
        hsparse = {
            'indices': sparse['indices'],
            'values':  [v * (1 - h) for v in sparse['values']]
        }
        hdense = [v * h for v in dense]
        return hdense, hsparse

    def display_pinecone_result(p_result):
        """
            Processes and formats the results from a Pinecone query for visualization.

            Args:
                p_result (dict): Results from Pinecone's query response.

            Returns:
                str: A formatted table of the query results using tabulate.
            """
        new_dict = []
        for idx, row in enumerate(p_result['matches']):
            new_dict.append({
                'l.p': idx + 1,
                'id': row['id'],
                'page': row['metadata']['page'],
                'paragraph': row['metadata']['paragraph'],
                'similarity_score': row['score']
            })
        df_results = pd.DataFrame(new_dict)
        text_wrap = df_results.apply(lambda col: col.map(lambda x:'\n'.join(textwrap.wrap(x, width=60)) if isinstance(x, str) else x))
        # results_textwrap = df_results.apply(lambda x: x.map('\n'.join(textwrap.wrap(x, width=60)) if isinstance(x, str) else x), axis=0)

        colalign = ['left','left','left','left','left']
        results_print = (tabulate(text_wrap, headers='keys', colalign=colalign, tablefmt='pretty'))
        return results_print

    # ==================================================================================================================








    # ==================================================================================================================

    # Define path to save processed data
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    saved_data_path = f"{base_name}_processed_document.pkl"

    # Check if processed data already exists
    if os.path.exists(saved_data_path):
        print(f"OCR document found, loading: {saved_data_path}")
        # Load from saved file
        with open(saved_data_path, "rb") as file:
            document = pickle.load(file)

    else:
        # Run OCR only if no data is already saved
        print("No saved data, running OCR...")
        document = process_document(
            loc_processor_id=processor_id,
            loc_project_id=project_id,
            loc_location=location,
            loc_file_path=file_path,
            loc_mime_type=mime_type,
            loc_processor_version="stable"
        )

        # Save processed data locally
        with open(saved_data_path, "wb") as file:
            pickle.dump(document, file)
            print(f"OCR Complete, and saved document as: {saved_data_path}")

    print("Detected languages on a first page:")

    for idx, page in enumerate(document.pages):
        # Iterate over all pages
        if page.detected_languages:  # Ensure there are detected languages
            for lang in page.detected_languages:
                if idx == 0:
                    print(f"page: {idx+1} {lang.language_code} ({lang.confidence:.1%} confidence)")


    block_list = create_dict(document)

    # Convert dict to pandas df
    df = pd.DataFrame([{
        'paragraph': block['paragraph'],
        'page': block['metadata']['page'],
        'location':block['metadata']['location']
    } for block in block_list
                      ])

    df.sort_values('page', ascending=True)
    df.reset_index(drop=True,inplace=True)

    # Create a copy of df for textwrap and tabulate.
    display_df = df.copy()
    display_df['location'] = display_df['location'].apply(format_location)

    # Running a tabulate function to get a preview of our paragraphs(choose iloc value for more rows).- Debugging line
    # result = tabulate_df_copy(display_df.iloc[:5])
    # print(result)

    # get the row count
    row_count = len(df)
    print(f'The row count is {row_count}')

    # Determine the longest paragraph
    len_list = []
    for idx, row in df.iterrows():
        text_len = (len((row['paragraph'])))
        len_list.append(int(text_len))

    print(f"The longest paragraph has: {max(len_list)} characters")

    # Run unittest for create embedding function.
    test_create_embeddings()

    # # Run the embeddings with a sample for display. - Debugging step (prints in a table results on a sample)
    # test_paragraphs_with_embeddings = create_embeddings(df,num_rows=5)
    #
    #
    # # Define the alignment.
    # colalign = ("center", "left", "left")
    #
    # # Tabulate:
    # test_paragraphs_with_embeddings_wrapped = [{key: wrap_truncate(value) for key, value in item.items()}
    #                                            for item in test_paragraphs_with_embeddings]
    # print(tabulate(test_paragraphs_with_embeddings_wrapped, headers='keys', tablefmt='simple',colalign=colalign))

    # Run the embedding with all rows.
    all_paragraphs_with_embeddings = create_embeddings(df, num_rows=row_count)
    print(f"The length of embedding list: {len(all_paragraphs_with_embeddings)}")
    # Create Pinecone index.
    pinecone_index = create_pinecone_index(index_name)

    # Print index stats.
    # print(pinecone_index.describe_index_stats())

    # Create train_df and train sparse encoder BM25.
    train_df = df.copy()
    enc = sparse_encoder(train_df)

    # Use reshaping function to create vectors to upload to Pinecone
    all_paragraphs_with_embeddings_to_upsert = create_vector_for_uploading(all_paragraphs_with_embeddings, enc)
    print(f"Vectors to upsert: {len(all_paragraphs_with_embeddings_to_upsert)}")

    # Use tabulate function for displaying vectors in a table form - Debuggung step: displays vectors with embeddings
    tabulate_vector = vector_preview(all_paragraphs_with_embeddings_to_upsert)
    # print(tabulate_vector)

    # Upsert our reshaped vectors,set batch_size=100, Pinecone will send them in chunks of 100 vectors at a time.
    pinecone_index.upsert(vectors=all_paragraphs_with_embeddings_to_upsert, batch_size=100)

    flag = True
    # Perform search query.
    while flag:

        search_query = input("Type what are you looking for: you can type contex and keywords: ")
        print("Type 'exit', or 'quit' to close application")
        if search_query.lower() in ['exit', 'quit']:
            print("See ya!")
            flag = False
            break
        else:
            search_query_as_vectors = convert_string_query_to_vectors(search_query)

            # Get the hybrid factor (h) from the user
            try:
                h = float(input("\nSet the hybrid search factor float(h) between 0.0 and 1.0 (for example 0.5)"
                                "\nThe result query will be a mixture of keyword and context."
                                "\n1.0 means 100% semantic search - application will search text for most relevant context,"
                                "\n0.0 means 100% keyword search - application will search for exact words in the text."
                                "\ntype search factor float(h): "))
            except ValueError:
                print("Invalid input for h. Must be a number between 0 and 1.")
                raise

            # Create a vector for a query
            hdense, hsparse = hybrid_search(search_query_as_vectors.get("dense"),
                                            search_query_as_vectors.get("sparse"), h)

            # Send query to database.
            pinecone_result = pinecone_index.query(
                top_k=3,
                vector=hdense,
                sparse_vector=hsparse,
                include_metadata=True
            )

            table = display_pinecone_result(pinecone_result)
            print(table)

if __name__ == '__main__':
    my_main()



