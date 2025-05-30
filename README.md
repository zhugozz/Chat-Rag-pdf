# 🤖📄 Chat RAG com PDF - Streamlit + OpenRouter

Este projeto implementa um sistema de **Perguntas e Respostas com base em PDFs** utilizando o conceito de **RAG (Retrieval-Augmented Generation)**. A aplicação permite que o usuário faça upload de um arquivo PDF, realize perguntas sobre seu conteúdo e receba respostas geradas por um modelo de linguagem da OpenRouter, com suporte de embeddings e busca semântica.

## 🚀 Tecnologias Utilizadas

- [Python 3.10+](https://www.python.org/)
- [Streamlit](https://streamlit.io/) - Interface web interativa
- [LangChain](https://www.langchain.com/) - Framework para aplicações de IA
- [Chroma](https://docs.trychroma.com/) - Banco de dados vetorial
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers) - Para geração de embeddings
- [OpenRouter](https://openrouter.ai/) - API para modelos de linguagem
- [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) - Carregamento e processamento de PDFs

## 📌 Lógica do Funcionamento

O sistema segue o fluxo do RAG:

1. **Upload do PDF**: O usuário carrega um arquivo PDF via interface Streamlit.
2. **Processamento do PDF**: O documento é lido pelo `PyPDFLoader` e dividido em páginas.
3. **Divisão em Chunks**: As páginas são fragmentadas em trechos menores com `RecursiveCharacterTextSplitter`.
4. **Geração de Embeddings**: Os trechos são convertidos em vetores usando embeddings da HuggingFace.
5. **Armazenamento Vetorial**: Os vetores são salvos com persistência no banco ChromaDB.
6. **Busca e Resposta**:
   - A pergunta do usuário é usada para recuperar os trechos mais relevantes via busca vetorial.
   - Os trechos relevantes e a pergunta são enviados ao modelo `openai/o4-mini` via OpenRouter.
   - O modelo gera uma resposta precisa com base no conteúdo do PDF.

## 🔐 Configuração da API (OpenRouter)

1. Crie uma conta gratuita em [OpenRouter](https://openrouter.ai).
2. Gere sua chave de API no painel da plataforma.
3. Configure a chave no código:

```python
import os
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-sua-chave-aqui"
```

## 🛠️ Instalação

Clone o repositório:
git clone https://github.com/seu-usuario/chat-rag-pdf.git
cd chat-rag-pdf


Crie e ative um ambiente virtual:
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows


Instale as dependências:
pip install -r requirements.txt


## ▶️ Execução do Projeto
```
Inicie a aplicação com:
streamlit run app.py
Acesse a interface no navegador (geralmente em http://localhost:8501).
```
---
## 📂 Estrutura do Projeto
```
chat-rag-pdf/
├── app.py                 # Código principal da aplicação
├── requirements.txt       # Dependências do projeto
├── README.md              # Documentação do projeto
└── text_index/            # Diretório para a base vetorial persistente (Chroma)
````
---
## 📄 Licença
Este projeto está licenciado sob os termos da MIT License.
Copyright © 2025 Hugo Leonardo
## 👨‍💻 Autor
Hugo Leonardo

GitHub: github.com/seu-usuario
LinkedIn: linkedin.com/in/seu-perfil

## 🎯 Objetivo
Este projeto foi criado para estudar e aplicar técnicas modernas de inteligência artificial, com foco em Retrieval-Augmented Generation (RAG) para processamento e interação com documentos PDF.
