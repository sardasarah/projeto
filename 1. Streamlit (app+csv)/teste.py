import pandas as pd

#Carrega o dataset
df = pd.read_csv("creditcard.csv")

#5 transações normais
normais = df[df["Class"] == 0].sample(5, random_state=42)

#5 transações fraudulentas
fraudes = df[df["Class"] == 1].sample(5, random_state=42)

#Junta tudo
teste = pd.concat([normais, fraudes])

#Embaralha as linhas
teste = teste.sample(frac=1, random_state=42)

#Remove a coluna alvo
teste_sem_classe = teste.drop(columns=["Class"])

#Salva o CSV para usar no Streamlit
teste_sem_classe.to_csv(
    "csv_teste_professora.csv",
    index=False
)
print("Arquivo criado com sucesso!")