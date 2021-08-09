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
import multiprocessing
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



@st.cache
def get_data(exemplos, option):
    return exemplos[option]

@st.cache
def fecth_data(table_name, sep=','):
    return pd.read_csv(f'dados/time series/{table_name}.csv', sep=sep)

@st.cache
def add_lags(df, output, lags):
    df = df.copy()
    for i in range(lags):
        df[f'x_{i+1}'] = df[output].shift(i + 1)
    df.dropna(axis=0, inplace=True)
    return df    

@st.cache(allow_output_mutation=True)
def pre_process_data(df, y_column):
    df = df.copy()
    # df.drop(columns_to_eliminate, axis=1, inplace=True)
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

@st.cache(allow_output_mutation=True)
def ModelLinearRegression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

@st.cache(allow_output_mutation=True)
def ModelPoliRegression(degree, X_train, y_train):
    model = make_pipeline(PolynomialFeatures(degree),LinearRegression())
    model.fit(X_train, y_train)
    return model

@st.cache(allow_output_mutation=True)
def ModelSVRLinear(X_train, y_train):
    model = SVR(kernel='linear')
    model.fit(X_train, y_train)
    return model

@st.cache(allow_output_mutation=True)
def ModelSVRPolynomial(C, degree, gamma, X_train, y_train):
    model = SVR(kernel='poly', C=C, degree=degree, gamma=gamma)
    model.fit(X_train, y_train)
    return model

@st.cache(allow_output_mutation=True)
def ModelSVRRBF(C, gamma, X_train, y_train):
    model = SVR(kernel='rbf', C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model

@st.cache(allow_output_mutation=True)
def ModelSVRSigmoid(C, gamma, X_train, y_train):
    model = SVR(kernel='sigmoid', C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model

@st.cache(allow_output_mutation=True)
def ModelDecisionTreeRegressor(criterion, splitter, max_depth, max_features, X_train, y_train):
    model = DecisionTreeRegressor(criterion=criterion, splitter=splitter, max_depth=max_depth, max_features=max_features)
    model.fit(X_train, y_train)
    return model

def autoregression():
    st.header('Séries Temporais')
    exemplos = { 
                # 'Imóveis': {'data': 'imóveis', 'sep': ','},
                'População': {'data': 'população', 'sep': ','},
                'Temperatura em Fortaleza': {'data': 'temperatura_fortaleza', 'sep': ','},
                'Temperatura na Índia': {'data': 'temperatura_india', 'sep': ','},
                'Temperatura em Punjab': {'data': 'temperatura_punjab', 'sep': ','},
                'Clima em Nova Deli': {'data': 'DailyDelhiClimateTrain', 'sep': ','}
                }
    option = st.selectbox('Escolha o exemplo', [key for key in exemplos])
    with st.expander('Dados'):
        linhas = st.slider('Número de linhas para exibir', value=5, max_value=100, min_value=5)
        df_temp = fecth_data(exemplos[option]['data'], sep=exemplos[option]['sep'])
        st.dataframe(df_temp.head(linhas))
        st.write(f'Número total de linhas = {df_temp.shape[0]}')
    with st.expander('Descrição'):
        
        if option == 'Carros':
            with open('markdowns/carros', 'r') as file:
                st.markdown(file.read())
                
    with st.expander('Gráfico'):
        tempo = st.selectbox('Escolha a variável de tempo', df_temp.columns, key='tempo-g')
        output = st.selectbox('Escolha a variável de output', df_temp.select_dtypes(exclude='object').columns, key='output-g')
        if st.checkbox('Categoria por cor'):
            color = st.selectbox('Variável para cor', 
                                 df_temp.drop([tempo, output], axis=1).select_dtypes(include='object').columns.to_list())
        else:
            color = None
        if st.button('Gerar gráfico'):
            if color:
                st.plotly_chart(px.line(df_temp.sort_values(tempo), x=tempo, y=output, color=color), use_container_width=True)
            else:
                st.plotly_chart(px.line(df_temp.sort_values(tempo), x=tempo, y=output), use_container_width=True)
    with st.expander('Ajustar modelo autoregressivo'):
        st.markdown('### Configuração do treinamento')
        train_size = st.slider('Proporção do conjunto de treinamento', value=0.7, min_value=0.5, max_value=0.9, step=0.05)
        st.markdown('### Configuração do método de autoregressão')
        metodo = st.selectbox('Escolha o método de autoregressão', 
                              ('Linear', 
                               'Polinomial', 
                               'Vetor de suporte linear', 
                               'Vetor de suporte polinomial', 
                               'Vetor de suporte rbf',
                               'Vetor de suporte sigmoid', 
                               'Árvore de decisão'))
        if metodo in ['Polinomial', 'Vetor de suporte polinomial']:
            degree = st.number_input('Grau do polinômio', value=2)
        if metodo in ['Vetor de suporte polinomial', 'Vetor de suporte rbf', 'Vetor de suporte sigmoid']:
            gamma = st.selectbox('Gama', ('scale', 'auto'))
            C = st.slider('Parâmetro de regularização C', value=1000, min_value=10, max_value=10000, step=10)
        if metodo == 'Árvore de decisão':
            criterion = st.selectbox('Critério de divisão dos ramos', ('mse', 'friedman_mse', 'mae'))
            splitter = st.selectbox('Divisor', ('best', 'random'))
            if st.checkbox('Profundidade customizada'):
                max_depth = st.number_input('Profundidade máxima da árvore', value=5, step=1, min_value=2)
            else:
                max_depth = None
            if st.checkbox('Limitar número de atributos'):
                max_features = st.selectbox('Tipo de limitação', ('auto', 'sqrt', 'log2'))
            else:
                max_features = None
        
        tempo = st.selectbox('Escolha a variável de tempo', df_temp.columns)
        output = st.selectbox('Escolha a variável de output', df_temp.select_dtypes(exclude='object').columns)
        exogeno = st.checkbox('Modelo exógeno')
        if exogeno:
            colunas = df_temp.drop([output, tempo], axis=1).columns.to_list()
            variaveis_exogenas = st.multiselect('Selecione as variáveis exógenas', colunas, 
                                                   default=colunas)
        
        lags = st.slider('Lag', min_value=1, max_value=df_temp.shape[0]//10, value=3)
        
        grafico = st.checkbox('Mostrar gráfico das previsões')
        if st.button('Calcular ajuste'):
            if exogeno:
                df_temp2 = df_temp[[tempo, output] + variaveis_exogenas].set_index(tempo).copy()
            else:
                df_temp2 = df_temp[[tempo, output]].set_index(tempo).copy()
            df_temp2 = add_lags(df_temp2, output, lags)
            X, y, onehot, mmscaler = pre_process_data(df_temp2, output)
            if X.shape[1] > 50:
                st.warning('Quantidade excessiva de variáveis após a transformação dos dados. Elime colunas categóricas que possuam muitos itens distintos e execute novamente o ajuste do modelo.')
            else:
                train_size = int(X.shape[0] * train_size)
                X_train = X[:train_size]
                X_test = X[train_size:]
                y_train = y[:train_size]
                y_test = y[train_size:]
                # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=train_size)
                if metodo == 'Linear':
                    model = ModelLinearRegression(X_train, y_train)
                elif metodo == 'Polinomial':
                    model = ModelPoliRegression(degree, X_train, y_train)
                elif metodo == 'Vetor de suporte linear':
                    model = ModelSVRLinear(X_train, y_train)
                elif metodo == 'Vetor de suporte polinomial':
                    model = ModelSVRPolynomial(C, degree, gamma, X_train, y_train)
                elif metodo == 'Vetor de suporte rbf':
                    model = ModelSVRRBF(C, gamma, X_train, y_train)
                elif metodo == 'Vetor de suporte sigmoid':
                    model = ModelSVRSigmoid(C, gamma, X_train, y_train)
                elif metodo == 'Árvore de decisão':
                    model = ModelDecisionTreeRegressor(criterion, splitter, max_depth, max_features, X_train, y_train)
                st.write('* Erro médio quadrático = ', mean_squared_error(y_test, model.predict(X_test)))
                st.write('* Score (treinamento)= ', r2_score(y_train, model.predict(X_train)))
                st.write('* Score (teste)= ', r2_score(y_test, model.predict(X_test)))
                if grafico:
                    result = df_temp2.copy()

                    result['Previsão'] = model.predict(X)
                    result = result[[output, 'Previsão']].stack().reset_index(name='Valor').rename(columns={'level_1': 'Tipo'})
                    st.plotly_chart(px.line(result, x=tempo, y='Valor', color='Tipo'), use_container_width=True)

    with st.expander('Fazer previsão'):
        
        if exogeno:
            inputs = {}
            for column in variaveis_exogenas:
                if column in df_temp.drop([output], axis=1).select_dtypes(exclude='object').columns:
                    inputs[column] = [st.slider(f'{column}', value=float(df_temp[column].median()), 
                                                min_value=float(df_temp[column].min()), 
                                                max_value=float(df_temp[column].max()))]
                else:
                    inputs[column] = [st.selectbox(f'{column}', df_temp[column].unique())]
        if st.button('Prever'):
            if exogeno:
                df_temp2 = df_temp[[tempo, output] + variaveis_exogenas].set_index(tempo).copy()
            else:
                df_temp2 = df_temp[[tempo, output]].set_index(tempo).copy()
            df_temp2 = add_lags(df_temp2, output, lags)
            X, y, onehot, mmscaler = pre_process_data(df_temp2, output)
            if X.shape[1] > 50:
                st.warning('Modelo não criado!')
            else:
                train_size = int(X.shape[0] * train_size)
                X_train = X[:train_size]
                X_test = X[train_size:]
                y_train = y[:train_size]
                y_test = y[train_size:]
                # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=train_size)
                if metodo == 'Linear':
                    model = ModelLinearRegression(X_train, y_train)
                elif metodo == 'Polinomial':
                    model = ModelPoliRegression(degree, X_train, y_train)
                elif metodo == 'Vetor de suporte linear':
                    model = ModelSVRLinear(X_train, y_train)
                elif metodo == 'Vetor de suporte polinomial':
                    model = ModelSVRPolynomial(C, degree, gamma, X_train, y_train)
                elif metodo == 'Vetor de suporte rbf':
                    model = ModelSVRRBF(C, gamma, X_train, y_train)
                elif metodo == 'Vetor de suporte sigmoid':
                    model = ModelSVRSigmoid(C, gamma, X_train, y_train)
                elif metodo == 'Árvore de decisão':
                    model = ModelDecisionTreeRegressor(criterion, splitter, max_depth, max_features, X_train, y_train)
                endo = df_temp.set_index(tempo).tail(lags).sort_index(ascending=False)
                endo = endo[output].values.reshape(1, -1)
                df_endo = pd.DataFrame(endo, columns=[f'x_{i + 1}' for i in range(lags)])
                if exogeno:
                    df_exo = pd.DataFrame(inputs)
                    df_input = pd.concat([df_exo, df_endo], axis=1)
                else:
                    df_input = df_endo
                X_bin = onehot.transform(df_input.select_dtypes(include=['object']))
                X_num = mmscaler.transform(df_input.select_dtypes(exclude=['object']))
                X_all = np.append(X_num, X_bin, axis=1)
                st.write(f"O valor previsto para o próximo período = ", round(model.predict(X_all)[0], 3))
            
        

