from llama_cloud_services import LlamaParse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import requests
import argparse
import glob
import os


def process_and_chunk_pdfs(input_folder: str, output_txt_folder: str, parser: LlamaParse, splitter: RecursiveCharacterTextSplitter):
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return []

    # Parse PDFs
    parsed_list = parser.parse(pdf_files)

    # Ensure output folder exists
    os.makedirs(output_txt_folder, exist_ok=True)
    page_seperator = "\n\n========== PAGE BREAK ==========\n\n"

    docs = []
    for i, parsed in enumerate(parsed_list):
        base_name = os.path.basename(pdf_files[i]).rsplit('.', 1)[0]
        txt_path = os.path.join(output_txt_folder, f"{base_name}.txt")

        # Save full markdown
        full_md = page_seperator.join(page.md for page in parsed.pages)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(full_md)
        print(f"Saved parsed text: {txt_path}")

        # Chunk pages
        for page_idx, page in enumerate(parsed.pages):
            chunks = splitter.split_text(page.md)
            for chunk in chunks:
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            'source': base_name,
                            'page': page_idx + 1,
                        }
                    )
                )
    return docs


def upload_docs(docs, api_url: str):

    payload = [
        {
            'page_content': doc.page_content,
            'metadata': doc.metadata
        }
        for doc in docs
    ]

    try:
        resp = requests.post(
            f"{api_url}/add_docs",
            json={'documents': payload}
        )
        resp.raise_for_status()
        print(f"Successfully uploaded {len(docs)} documents.")
    except Exception as e:
        print(f"Failed to upload documents: {e}")


def main():
    parser_arg = argparse.ArgumentParser(description="Parse, chunk PDFs and upload to vector DB via add_docs API.")
    parser_arg.add_argument(
        'folder',
        help='Path to folder containing PDF files'
    )
    parser_arg.add_argument(
        '--output', '-o',
        default='rag/papers/parsed_txt',
        help='Folder to save parsed text files'
    )
    parser_arg.add_argument(
        '--api-url', '-u',
        default='http://localhost:8000',
        help='Base URL of the app API'
    )
    args = parser_arg.parse_args()

    load_dotenv()
    os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv('LLAMA_CLOUD_API_KEY')

    # Initialize parser and splitter
    llama_parser = LlamaParse(
        auto_mode=True,
        auto_mode_trigger_on_image_in_page=True,
        auto_mode_trigger_on_table_in_page=True,
        split_by_page=True,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n"]
    )

    docs = process_and_chunk_pdfs(
        args.folder,
        args.output,
        llama_parser,
        text_splitter
    )
    if docs:
        upload_docs(docs, args.api_url)


if __name__ == '__main__':
    main()
