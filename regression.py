import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



@st.cache
def get_data(exemplos, option):
    return exemplos[option]

@st.cache
def fecth_data(table_name, sep=','):
    return pd.read_csv(f'dados/{table_name}.csv', sep=sep)

@st.cache(allow_output_mutation=True)
def pre_process_data(df, y_column, columns_to_eliminate):
    df = df.copy()
    df.drop(columns_to_eliminate, axis=1, inplace=True)
    y = df[y_column]
    X = df.drop(y_column, axis=1)
    onehot = OneHotEncoder(sparse=False, drop="first")
    try:
        X_bin = onehot.fit_transform(X.select_dtypes(include=['object']))
    except:
        X_bin = np.array([[] for _ in range(X.shape[0])])
    mmscaler = MinMaxScaler()
    try:
        X_num = mmscaler.fit_transform(X.select_dtypes(exclude=['object']))
    except:
        X_num = np.array([[] for _ in range(X.shape[0])])
    X_all = np.append(X_num, X_bin, axis=1)
    return X_all, y, onehot, mmscaler

def regression():
    st.header('Regressão')
    exemplos = { 
                'AirBnb Nova Iorque': {'data': 'airbnb_ny2', 'sep': ','},
                'Alugueis em São Paulo': {'data': 'alugueis_brasil', 'sep': ','},
                'Alugueis no Brasil': {'data': 'alugueis_brasil2', 'sep': ','},
                'Cancer de mama': {'data': 'Breast_cancer_data', 'sep': ','},
                'Dados demográficos': {'data': 'Country-data', 'sep': ','},
                'Banco da alemanha': {'data': 'german_credit_data', 'sep': ','},
                'Carros': {'data': 'carros', 'sep': ','}
                }
    option = st.selectbox('Escolha o exemplo', [key for key in exemplos])
    with st.beta_expander('Dados'):
        linhas = st.slider('Número de linhas para exibir', value=5, max_value=100, min_value=5)
        df_temp = fecth_data(exemplos[option]['data'], sep=exemplos[option]['sep'])
        st.dataframe(df_temp.head(linhas))
        st.write(f'Número total de linhas = {df_temp.shape[0]}')
    with st.beta_expander('Descrição'):
        
        if option == 'Carros':
            with open('markdowns/carros', 'r') as file:
                st.markdown(file.read())
        # elif option == 'Alugueis em São Paulo':
        #     with open('markdowns/alugueis_brasil', 'r') as file:
        #         st.markdown(file.read())
        # elif option == 'Alugueis no Brasil':
        #     with open('markdowns/alugueis_brasil2', 'r') as file:
        #         st.markdown(file.read())
        # elif option == 'Banco':
        #     with open('markdowns/banco', 'r') as file:
        #         st.markdown(file.read())
    with st.beta_expander('Ajustar modelo de regressão'):
        st.markdown('### Configuração do treinamento')
        train_size = st.slider('Proporção do conjunto de treinamento', value=0.7, min_value=0.5, max_value=0.9, step=0.05)
        st.markdown('### Configuração do método de classificação')
        metodo = st.selectbox('Escolha o método de regressão', ('linear', 'polinomial'))
        if metodo == 'polinomial':
            grau = st.number_input('Grau do polinômio', value=2)
        
        output = st.selectbox('Escolha a variável de output', df_temp.select_dtypes(exclude='object').columns)
        tipo = st.radio('Tipo de escolha dos inputs', ('incluir', 'eliminar'))
        if tipo == 'incluir':
            colunas = df_temp.drop(output, axis=1).columns.to_list()
            colunas_para_incluir = st.multiselect('Selecione as colunas para incluir', colunas, default=colunas)
            colunas_para_eliminar = [value for value in df_temp.drop(output, axis=1).columns if value not in colunas_para_incluir]
        else:
            colunas_para_eliminar = st.multiselect('Selecione as colunas para eliminar', df_temp.drop(output, axis=1).columns)
        
        if st.button('Calcular ajuste'):
            X, y, onehot, mmscaler = pre_process_data(df_temp, output, colunas_para_eliminar)
            if metodo == 'linear':
                model = LinearRegression()
            elif metodo == 'polinomial':
                model = make_pipeline(PolynomialFeatures(grau),LinearRegression())
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=train_size)
            model.fit(X_train, y_train)
            st.write('* Erro médio quadrático = ', mean_squared_error(y_test, model.predict(X_test)))
            st.write('* Score = ', r2_score(y_test, model.predict(X_test)))

    with st.beta_expander('Fazer previsão'):
        inputs = {}
        for column in df_temp.drop([output] + colunas_para_eliminar, axis=1).columns:
            if column in df_temp.drop([output] + colunas_para_eliminar, axis=1).select_dtypes(exclude='object').columns:
                inputs[column] = [st.number_input(f'{column}', value=df_temp[column].median())]
            else:
                inputs[column] = [st.selectbox(f'{column}', df_temp[column].unique())]
        if st.button('Prever'):
            X, y, onehot, mmscaler = pre_process_data(df_temp, output, colunas_para_eliminar)
            if metodo == 'linear':
                model = LinearRegression()
            elif metodo == 'polinomial':
                model = make_pipeline(PolynomialFeatures(grau),LinearRegression())
            model.fit(X, y)
            df_input = pd.DataFrame(inputs)
            X_bin = onehot.transform(df_input.select_dtypes(include=['object']))
            X_num = mmscaler.transform(df_input.select_dtypes(exclude=['object']))
            X_all = np.append(X_num, X_bin, axis=1)
            st.write(f"O valor previsto para a variável '{output}' = ", round(model.predict(X_all)[0], 3))
            
        
