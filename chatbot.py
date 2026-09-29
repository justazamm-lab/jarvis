from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from fastapi import FastAPI,Request

app = FastAPI()

from fastapi.staticfiles import StaticFiles
from fastapi.responses  import FileResponse

from pathlib import Path
BASE_DIR = Path(__file__).parent

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(BASE_DIR / "static" / "index.html")

load_dotenv()

llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    temperature=0
)

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

embedding = FastEmbedEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2",
  cache_dir=str(BASE_DIR / "fastembed_cache")
)

vectordb = Chroma(
  persist_directory=str(BASE_DIR / "chromadb"),
  embedding_function=embedding
)

retriver = vectordb.as_retriever(search_kwargs={"k":3})


from langchain_community.document_loaders import PyPDFLoader,TextLoader,CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil
import tempfile
from fastapi import UploadFile,File,HTTPException


splitter = RecursiveCharacterTextSplitter(
  chunk_size = 1000,
  chunk_overlap=200
)

loader_map = {
  ".pdf":PyPDFLoader,
  ".csv":CSVLoader,
  ".txt":TextLoader
}

@app.post("/UCHAT/INGEST")
def fileingest(
    request: Request,
    file: UploadFile = File(...)
):

    try:

        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

        ext = os.path.splitext(file.filename)[1].lower()

        loader_cls = loader_map.get(ext)

        if loader_cls is None:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: {ext}"
            )

        content_length = request.headers.get("content-length")

        if (
            content_length
            and int(content_length) > MAX_FILE_SIZE
        ):
            raise HTTPException(
                status_code=413,
                detail="File size exceeds 10 MB limit."
            )

        file_size = 0

        while chunk := file.file.read(1024 * 1024):

            file_size += len(chunk)

            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail="File size exceeds 10 MB limit."
                )

        file.file.seek(0)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext
        ) as tmp:

            shutil.copyfileobj(
                fsrc=file.file,
                fdst=tmp
            )

            tmp_path = tmp.name

        try:

            loader = loader_cls(tmp_path)

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file.filename

            chunks = splitter.split_documents(docs)

            ids = vectordb.add_documents(chunks)

            total_count = vectordb._collection.count()

            return {
                "filename": file.filename,
                "file_size_mb": round(
                    file_size / (1024 * 1024),
                    2
                ),
                "chunks_added": len(chunks),
                "ids": ids,
                "total_chunks_in_db": total_count
            }

        finally:

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )





from pydantic import BaseModel
from typing import List,Optional,Dict,Any



from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from fastapi import HTTPException

class Chatrequest(BaseModel):
  msg:str 
  history: Optional[List[dict]] = []

class AIResponse(BaseModel):
  sources: List[Dict[str,Any]]
  reply: str
  


from langchain_core.messages import AIMessage,HumanMessage,SystemMessage


chat_llm = init_chat_model(
   model="openai/gpt-oss-120b",
   model_provider="groq",
   temperature=0.7
)

from langchain.messages import trim_messages



@app.post("/UCHAT/Chatbot",response_model=AIResponse)

def chatbot(request: Chatrequest):
 try:

   docs = retriver.invoke(request.msg)

   context = "\n\n".join(doc.page_content for doc in docs)


   messages = [SystemMessage(content=f"""
You are Jarvis.

Rules:
- Match the users intent.
- If they ask a question, answer directly.
- If they ask for a summary, summarize.
- If they ask for extraction, extract.
- If they ask for comparison, compare.
- If they ask for steps, provide steps.
- Use context only when relevant.
- Do not unnecessarily summarize everything.


context:
{context}""")]

   for turn in request.history :
     if turn.get("role") == "user":
       messages.append(HumanMessage(content=turn.get("content","")))
     elif turn.get("role") == 'assistant':
       messages.append(AIMessage(content=(turn.get("content",""))))

   messages.append(HumanMessage(content=(request.msg)))

   trimmed = trim_messages(messages,
                           max_tokens=11,
                           token_counter=len,
                           strategy="last",
                           start_on="human",
                           include_system=True)

   response = chat_llm.invoke(trimmed).content

   return AIResponse(reply=response, sources=[{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs] )

 
 except Exception as e:
   raise HTTPException(status_code=500,detail=str(e))

