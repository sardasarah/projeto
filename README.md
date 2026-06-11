# Detecção de Fraude em Cartões de Crédito

Sistema de detecção de transações fraudulentas em cartões de crédito utilizando técnicas de Machine Learning aplicadas a um dataset altamente desbalanceado.

---

## Integrantes

| Nome | RA |
|---|---|
| Amanda Villar Moura | 202327-8 |
| Isabella Machado de Souza Duarte | 203110-7 |
| Sarah Cristina Pereira Costa | 204136-5 |

---

## Descrição do Problema

Fraudes em cartões de crédito causam bilhões em prejuízos financeiros anualmente. O desafio é identificar transações fraudulentas em tempo real, considerando que as fraudes são eventos raros — representando apenas **0,17%** do total de transações — o que caracteriza um problema de classes altamente desbalanceadas.

---

## Objetivo do Projeto

Desenvolver um modelo de Machine Learning capaz de classificar transações como legítimas ou fraudulentas, maximizando a detecção de fraudes (Recall) sem comprometer excessivamente a experiência do usuário com bloqueios indevidos (Precisão).

---

## Dataset Utilizado

- **Nome:** Credit Card Fraud Detection
- **Fonte:** [Kaggle — mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Registros:** 286.784 transações
- **Variáveis:** 31 colunas (Time, V1–V28, Amount, Class)
- **Desbalanceamento:** ~0,17% de fraudes (492 casos)

> As features V1–V28 foram previamente transformadas por PCA por questões de confidencialidade.

---

## Tipo de Problema de Machine Learning

**Classificação Binária Supervisionada com Classes Desbalanceadas**

- Classe 0: Transação legítima
- Classe 1: Transação fraudulenta

---

## Metodologia

1. **Análise Exploratória (EDA):** distribuição das classes, análise de Amount e Time por classe, estatísticas descritivas das features V1–V28 e mapa de correlação.

2. **Pré-processamento:**
   - Transformação logarítmica em Amount: `log(Amount + 1)`
   - Padronização com StandardScaler
   - Remoção das colunas Time, Amount e Amount_log
   - Divisão estratificada: 60% treino / 20% validação / 20% teste

3. **Tratamento do desbalanceamento:** aplicação de **SMOTE** exclusivamente no conjunto de treino, criando amostras sintéticas da classe minoritária por interpolação entre exemplos reais.

4. **Treinamento e avaliação:** três classificadores treinados com validação cruzada Stratified K-Fold (k=5), usando AUC-PR como métrica principal.

5. **Otimização de threshold:** uso do conjunto de validação para encontrar o threshold que maximiza o F1-Score de cada modelo, aplicado posteriormente no conjunto de teste.

---

## Modelos Treinados

| Modelo | Observações |
|---|---|
| Regressão Logística | `class_weight='balanced'`, `max_iter=1000` |
| Random Forest | `class_weight='balanced'`, `n_estimators=100` |
| AdaBoost | `n_estimators=100`, balanceamento via SMOTE |

---

## Modelo Final Escolhido

**Random Forest**

Apresentou o melhor desempenho em **AUC-PR** — a métrica mais informativa para dados desbalanceados — combinando alta capacidade de detecção de fraudes com Precisão razoável. Adicionalmente, permite a análise de importância das features, agregando interpretabilidade ao modelo.

---

## Métricas de Avaliação

- **AUC-PR** (principal): área sob a curva Precisão-Recall — foca no desempenho sobre a classe minoritária
- **Recall:** proporção de fraudes reais corretamente detectadas — métrica mais crítica do problema
- **F1-Score:** média harmônica entre Precisão e Recall
- **AUC-ROC:** área sob a curva ROC
- **Precisão e Acurácia:** métricas complementares

> A acurácia não é utilizada como métrica principal pois é enganosa em datasets desbalanceados: um modelo que sempre prediz "legítima" atingiria 99,83% de acurácia sem detectar nenhuma fraude.

---

## Principais Resultados

> Preencher com os valores obtidos após execução final do notebook.

| Classificador | Acurácia | Precisão | Recall | F1-Score | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|---|
| Regressão Logística | 0.9729 | 0.0539 | 0.8980 | 0.1017 | 0.9703 | 0.6850 |
| Random Forest | 0.9995 | 0.9277 | 0.7857 | 0.8508 | 0.9665 | 0.8370 |
| AdaBoost | 0.9754 | 0.0597 | 0.9082 | 0.1120 | 0.9769 | 0.7485 |

---

## Estrutura dos Arquivos

```
deteccao-fraude/
│
├── 1. Streamlit/                 # Aplicação Streamlit
│   └── app.py 
│   └── gerador_de_arquivos.py    # Dados de entrada 
│                       
├── requirements.txt              # Dependências do projeto
├── README.md                     # Documentação do projeto
│
├── 4. Notebooks/
│   └── notebook_atualizado.ipynb # Notebook revisado (P2)
│
├── 5. Model/
│   ├── modelo_final.joblib       # Modelo Random Forest treinado
│   └── scaler.joblib             # StandardScaler para pré-processamento
│
├── 6. Reports/
│   └── relatorio_atualizado.pdf  # Relatório final
│
└── 7. Data/
    └── LinkCSV.txt               # Dataset de treinamento
```

---

## Tecnologias Utilizadas

- Python 3.10+
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn
- joblib
- Streamlit
- Google Colab (desenvolvimento do notebook)

---

## Instruções para Executar o Notebook

1. Acesse [Google Colab]([https://colab.research.google.com/](https://colab.research.google.com/drive/1aNnaDTUhD5qdjqC6ne4hG3j9QrhV7Wgb?usp=sharing)) e faça upload do arquivo 

2. Faça o download do dataset no Kaggle:
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud
   ```
   Ou acesse diretamente: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

3. Faça upload do arquivo `creditcard.csv` no Colab (ícone de pasta na barra lateral)

4. Instale a dependência adicional:
   ```python
   !pip install imbalanced-learn
   ```

5. Execute todas as células em ordem (`Runtime > Run all`)

6. Ao final, os arquivos `modelo_final.joblib` e `scaler.joblib` serão gerados e podem ser baixados para uso no app

---

## Instruções para Executar o App Streamlit

### Localmente

1. Clone o repositório:
   ```bash
   git clone https://github.com/chadommas/Deteccao_de_Fraudes_P2
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Certifique-se de que os arquivos `modelo_final.joblib` e `scaler.joblib` estão na pasta `model/`

4. Execute o app:
   ```bash
   streamlit run app.py
   ```

### Utilizando o App

1. Acesse a aplicação pelo link publicado (abaixo)
2. Faça upload de um arquivo CSV com transações no formato do dataset original (colunas V1–V28 e Amount)
3. Clique em **Analisar**
4. Visualize a tabela com as predições e as probabilidades de fraude por transação

---

## Link do App Publicado

🔗 (https://deteccaodefraude.streamlit.app/)

---

## Limitações

- As features V1–V28 são componentes de PCA — não há interpretação direta do significado de cada variável
- O modelo foi treinado com dados de transações europeias de 2013; pode não generalizar bem para outros contextos geográficos ou períodos
- O SMOTE gera amostras sintéticas por interpolação, o que pode não representar fielmente o comportamento real de novos tipos de fraude
- O app requer que o CSV de entrada esteja no mesmo formato do dataset original do Kaggle

---

## Conclusão

O projeto demonstrou que é possível construir um sistema eficaz de detecção de fraudes mesmo diante de um desbalanceamento extremo de classes. As principais contribuições da versão P2 foram a incorporação do SMOTE para tratamento adequado do desbalanceamento em todos os modelos (especialmente o AdaBoost) e o uso do conjunto de validação para otimização do threshold de decisão, resultando em melhor equilíbrio entre Precisão e Recall.

O modelo Random Forest foi escolhido como solução final por apresentar o melhor AUC-PR, sendo implantado em uma aplicação Streamlit que simula o uso real da ferramenta por analistas financeiros.
