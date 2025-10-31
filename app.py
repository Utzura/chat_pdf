import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# Título con estilo
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Generación Aumentada por Recuperación (RAG) 💬</h1>",
    unsafe_allow_html=True
)
st.write("Versión de Python:", platform.python_version())
st.markdown("---")

# Load and display image
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar con info más clara
with st.sidebar:
    st.title("📝 RAG Asistente")
    st.write("Este agente te ayudará a analizar PDFs usando OpenAI y embeddings.")

# Get API key from user
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# PDF uploader
st.subheader("📄 Cargar PDF")
pdf = st.file_uploader("Selecciona un archivo PDF", type="pdf")

# Process the PDF if uploaded
if pdf is not None and ke:
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos ✅")
        
        # Create embeddings and knowledge base
        with st.spinner("Creando base de conocimiento..."):
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
        st.success("Base de conocimiento creada ✅")
        
        # User question interface with expander
        st.subheader("💡 Haz tu pregunta sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")
        
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            
            with st.expander("Ver respuesta"):
                st.markdown(response)
                
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("📄 Por favor carga un archivo PDF para comenzar")
