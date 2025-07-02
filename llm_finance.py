import ofxparse
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

# =============== LEITURA DE OFX ===============
df = pd.DataFrame()

extratos_dir = Path("extratos")
extratos_dir.mkdir(exist_ok=True)  # cria pasta se não existir

arquivos_ofx = [f for f in extratos_dir.iterdir() if f.suffix.lower() == ".ofx"]

if not arquivos_ofx:
    raise FileNotFoundError("Nenhum arquivo OFX encontrado na pasta 'extratos'.")

for arquivo in arquivos_ofx:
    with open(arquivo, "r", encoding="utf-8") as ofx_file:
        ofx = ofxparse.OfxParser.parse(ofx_file)

    transactions_data = []
    for account in ofx.accounts:
        for transaction in account.statement.transactions:
            transactions_data.append({
                "Data": transaction.date,
                "Valor": transaction.amount,
                "Descrição": transaction.memo,
                "ID": transaction.id,
            })

    df_temp = pd.DataFrame(transactions_data)
    df_temp["Data"] = df_temp["Data"].apply(lambda x: x.date())
    df_temp["Valor"] = df_temp["Valor"].astype(float)
    df = pd.concat([df, df_temp])

df = df.drop_duplicates(subset="ID").set_index("ID")
df = df.sort_values(by="Data")

# =============== LLM CATEGORIZAÇÃO ===============
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import time

_ = load_dotenv(find_dotenv())

template = """
Você é um analista de dados, trabalhando em um projeto de limpeza de dados.
Seu trabalho é escolher uma categoria adequada para cada lançamento financeiro
que vou te enviar.

Todos são transações financeiras de uma pessoa física.

Escolha uma dentre as seguintes categorias:
- Alimentação
- Receitas
- Saúde
- Mercado
- Educação
- Compras
- Transporte
- Investimento
- Transferências para terceiros
- Telefone
- Moradia

Escolha a categoria deste item:
{text}

Responda apenas com a categoria.
"""

prompt = PromptTemplate.from_template(template)
chat = ChatGroq(model="llama-3.3-70b-versatile")
chain = prompt | chat | StrOutputParser()

# processa item a item com delay para não estourar rate limit
categorias = []
for desc in df["Descrição"]:
    categoria = chain.invoke(desc)
    categorias.append(categoria)
    time.sleep(2)  # espera 2 segundos entre requests para não ultrapassar limite

df["Categoria"] = categorias

# =============== FILTRO E SALVA CSV ===============
df = df[df["Data"] >= datetime(2024, 3, 1).date()]
df.to_csv("finances.csv", encoding="utf-8", index=True)

print("✅ finances.csv criado com sucesso!")

