import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
import os
import tempfile
import unicodedata
import traceback

# Configuração da API do OpenRouter
os.environ["OPENROUTER_API_KEY"] = "sk-or-sua-chave"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# --- Sidebar ---
st.sidebar.title("Configurações")
uploaded_file = st.sidebar.file_uploader("Carregar PDF", type=["pdf"])
persist_dir = "text_index"
collection_name = "dcd_store"

# --- Título da Página ---
st.title("Chat com - RAG PDF 🤖📄")

# --- Modelos e embeddings ---
if "embeddings_model" not in st.session_state:
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    st.session_state.embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

embeddings_model = st.session_state.embeddings_model

# --- Carrega PDF e indexa ---
def process_pdf_and_index(pdf_file):
    if not pdf_file:
        st.warning("Por favor, carregue um arquivo PDF.")
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        pdf_path = tmp_file.name

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            return Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings_model,
                collection_name=collection_name
            )
        except Exception as e:
            st.error(f"Erro ao carregar o banco Chroma: {str(e).encode('latin1', errors='ignore').decode('latin1')}")
            return None

    try:
        loader = PyPDFLoader(pdf_path, extract_images=False)
        pages = loader.load_and_split()

        # Limpa possíveis problemas de encoding nos conteúdos das páginas
        for page in pages:
            page.page_content = page.page_content.encode('latin1', errors='ignore').decode('latin1')

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        pages_splitted = splitter.split_documents(pages)

        db = Chroma.from_documents(
            documents=pages_splitted,
            embedding=embeddings_model,
            persist_directory=persist_dir,
            collection_name=collection_name
        )
        return db
    except Exception as e:
        st.error(f"Erro ao processar o PDF ou criar o banco Chroma: {str(e).encode('latin1', errors='ignore').decode('latin1')}")
        return None
    finally:
        os.unlink(pdf_path)

# --- Inicializar banco vetorial ---
vectordb = process_pdf_and_index(uploaded_file)
retriever = vectordb.as_retriever(search_kwargs={"k": 1}) if vectordb else None

# --- Interface principal ---
question = st.text_input("Digite sua pergunta:")

if question and retriever:
    with st.spinner("Buscando resposta..."):
        try:
            docs = retriever.invoke(question)

            # Corrige encoding nos conteúdos dos documentos
            context = "\n\n".join(
                doc.page_content.encode('latin1', errors='ignore').decode('latin1')
                for doc in docs
            )[:3000]

            response = client.chat.completions.create(
                model="openai/o4-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um assistente especialista em analisar documentos em PDF."
                    },
                    {
                        "role": "user",
                        "content": f"Contexto:\n{context}\n\nPergunta: {question}"
                    }
                ],
                extra_headers={
                    "HTTP-Referer": "https://sua-loja.com",
                    "X-Title": "Chat-RAG-Constituicao"
                },
                max_tokens=300,
                temperature=0.7
            )

            # Corrige encoding na resposta do modelo
            output = response.choices[0].message.content.encode("latin1", errors="ignore").decode("latin1")

            st.markdown("### Resposta:")
            st.write(output)

        except Exception as e:
            st.error("Erro ao consultar o modelo:")
            st.code(traceback.format_exc(), language="python")
else:
    if question and not retriever:
        st.error("Banco vetorial não inicializado. Verifique se o PDF foi carregado corretamente.")
