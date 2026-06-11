#Importa as bibliotecas que serão utilizadas no projeto
import streamlit as st      #Criação da interface web
import pandas as pd         #Manipulação de tabelas (DataFrames)
import numpy as np          #Operações matemáticas
import joblib              #Carregar modelo treinado e scaler


#Define o título, ícone e layout da aplicação
st.set_page_config(
    page_title="Detector de Fraudes",
    page_icon="💳",
    layout="wide"
)

# @st.cache_resource faz com que o modelo seja carregado
# apenas uma vez, evitando lentidão
@st.cache_resource
def carregar_modelo():

    #Carrega o modelo treinado
    modelo = joblib.load("modelo_final.joblib")

    #Carrega o scaler usado durante o treinamento
    scaler = joblib.load("scaler.joblib")

    return modelo, scaler

#Executa a função e guarda o modelo e o scaler em variáveis
modelo, scaler = carregar_modelo()

#Esta função recebe um DataFrame e devolve
#as previsões realizadas pelo modelo

def realizar_predicao(df):
    #Faz uma cópia dos dados para não alterar o original
    dados = df.copy()

    #Aplica transformação logarítmica na coluna Amount
    #para reduzir a influência de valores muito altos
    dados["Amount_log"] = np.log1p(
        dados["Amount"]
    )

    #Aplica a padronização usando o scaler treinado
    dados["Amount_scaled"] = scaler.transform(
        dados[["Amount_log"]]
    ).flatten()

    #Lista das variáveis utilizadas pelo modelo
    features = [
        'V1','V2','V3','V4','V5','V6','V7',
        'V8','V9','V10','V11','V12','V13',
        'V14','V15','V16','V17','V18','V19',
        'V20','V21','V22','V23','V24','V25',
        'V26','V27','V28','Amount_scaled'
    ]

    #Seleciona apenas as colunas usadas na previsão
    X = dados[features]

    #Retorna a probabilidade de cada registro ser fraude
    probabilidades = modelo.predict_proba(X)[:, 1]

    #Retorna a classificação final (0 ou 1)
    predicoes = modelo.predict(X)

    #Cria um DataFrame para exibir os resultados
    resultado = df.copy()

    #Converte a probabilidade para porcentagem
    resultado["Probabilidade_Fraude (%)"] = (
        probabilidades * 100
    ).round(2)

    #Se a previsão for 1 será fraude
    #Se a previsão for 0 será normal
    resultado["Classificacao"] = np.where(
        predicoes == 1,
        "🚨 Fraude",
        "✅ Normal"
    )

    #Retorna os resultados
    return resultado, predicoes

#Título principal da página
st.title("💳 Sistema de Detecção de Fraudes")
#Linha divisória visual
st.divider()
#Campo para upload do arquivo CSV

arquivo = st.file_uploader(
    "Selecione um arquivo CSV",
    type=["csv"]
)
#Verifica se o usuário enviou um arquivo
if arquivo is not None:
    try:
        #Lê o CSV enviado
        df = pd.read_csv(arquivo)
        #Exibe uma prévia das primeiras linhas
        st.subheader("Pré-visualização dos Dados")
        st.dataframe(
            df.head(),
            use_container_width=True
        )
        #Mostra quantos registros existem
        st.info(
            f"{len(df)} registros encontrados."
        )
        #Botão que inicia a análise
        if st.button(
            "🔍 Analisar Transações",
            use_container_width=True
        ):
            #Lista das colunas obrigatórias
            colunas_necessarias = [
                f"V{i}" for i in range(1, 29)
            ] + ["Amount"]
            #Verifica quais colunas estão faltando
            faltando = [
                c for c in colunas_necessarias
                if c not in df.columns
            ]
            #Se faltar alguma coluna, mostra erro
            if faltando:
                st.error(
                    f"Colunas ausentes: {', '.join(faltando)}"
                )
            else:
                #Executa a predição
                resultado, predicoes = realizar_predicao(df)
                #Soma todas as fraudes encontradas
                total_fraudes = int(predicoes.sum())
                #Cria 3 colunas para métricas
                col1, col2, col3 = st.columns(3)
                #Total de registros analisados
                col1.metric(
                    "Registros",
                    len(resultado)
                )
                #Total de fraudes encontradas
                col2.metric(
                    "Fraudes",
                    total_fraudes
                )
                #Total de registros normais
                col3.metric(
                    "Normais",
                    len(resultado) - total_fraudes
                )
                st.divider()
                #Título da seção de resultados
                st.subheader("Resultado da Análise")
                #Exibe somente as colunas mais importantes
                st.dataframe(
                    resultado[
                        [
                            "Amount",
                            "Probabilidade_Fraude (%)",
                            "Classificacao"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )
                #Converte o resultado para CSV
                csv = resultado.to_csv(
                    index=False
                ).encode("utf-8")
                #Botão para baixar os resultados
                st.download_button(
                    label="⬇️ Baixar Resultado",
                    data=csv,
                    file_name="resultado_fraudes.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    #Captura possíveis erros
    except Exception as erro:
        st.error(
            f"Erro ao processar o arquivo: {erro}"
        )
#Caso nenhum arquivo tenha sido enviado
else:
    st.info(
        "Faça o upload de um arquivo CSV para iniciar a análise."
    )