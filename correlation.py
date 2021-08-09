import streamlit as st
import pandas as pd
import plotly.express as px



px.set_mapbox_access_token('pk.eyJ1Ijoicm9uYWxkb2xhZ2UiLCJhIjoiY2tldm5yMnl1MGNxcTJ6bm9qcnR6eDl0ZSJ9.SHArbKTHFkYTNFaVro_gAw')

@st.cache
def get_data(exemplos, option):
    return exemplos[option]

@st.cache
def fecth_data(table_name, sep=','):
    return pd.read_csv(f'dados/{table_name}.csv', sep=sep)

def correlation():
    st.header('Análise de correlação')
    exemplos = { 
                'AirBnb Nova Iorque': {'data': 'airbnb_ny2', 'sep': ','},
                'Alugueis em São Paulo': {'data': 'alugueis_brasil', 'sep': ','},
                'Alugueis no Brasil': {'data': 'alugueis_brasil2', 'sep': ','},
                'Cancer de mama': {'data': 'Breast_cancer_data', 'sep': ','},
                'Dados demográficos': {'data': 'Country-data', 'sep': ','},
                'Banco da alemanha': {'data': 'german_credit_data', 'sep': ','},
                'Carros': {'data': 'CarPrice_Assignment', 'sep': ','}
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
    # if option == 'AirBnb Nova Iorque':
        #     with open('markdowns/airbnb_ny', 'r') as file:
        #         st.markdown(file.read())
        # if option == 'AirBnb Nova Iorque':
        #     with open('markdowns/airbnb_ny', 'r') as file:
        #         st.markdown(file.read())
        # elif option == 'Alugueis em São Paulo':
        #     with open('markdowns/alugueis_brasil', 'r') as file:
        #         st.markdown(file.read())
        # elif option == 'Alugueis no Brasil':
        #     with open('markdowns/alugueis_brasil2', 'r') as file:
        #         st.markdown(file.read())
        # elif option == 'Banco':
        #     with open('markdowns/banco', 'r') as file:
        #         st.markdown(file.read())
    with st.expander('Correlação'):
        metodo = st.selectbox('Escolha o coeficiente de correlação', ('pearson', 'spearman', 'kendall'))
        colunas_para_eliminar = st.multiselect('Selecione as colunas para eliminar', df_temp.columns)
        if st.button('Calcular'):
            df = df_temp.drop(colunas_para_eliminar, axis=1).corr(method=metodo)
            st.dataframe(df)
    with st.expander('Correlação agrupada'):
        metodo = st.selectbox('Escolha o coeficiente de correlação', ('pearson', 'spearman', 'kendall'), key='grupo')
        colunas_de_agrupamento = st.multiselect('Selecione as colunas para eliminar', df_temp.select_dtypes(include='object').columns)
        if st.button('Calcular', key='calc-grupo'):
            df = df_temp.groupby(colunas_de_agrupamento).corr(method=metodo)
            st.dataframe(df)
        # colunas_para_eliminar = st.multiselect('Selecione as colunas para eliminar', df_temp.columns)
    with st.expander('Gráficos'):
        x = st.selectbox('Escolha a variável para o eixo x', df_temp.select_dtypes(exclude='object').columns)
        y = st.selectbox('Escolha a variável para o eixo y', df_temp.select_dtypes(exclude='object').columns)
        if st.checkbox('cor'):
            color = st.selectbox('', df_temp.columns, key='color')
        else:
            color=None
        if st.checkbox('tamanho'):
            size = st.selectbox('', df_temp.columns, key='size')
        else:
            size=None
        if st.checkbox('forma'):
            forma = st.selectbox('', df_temp.columns, key='forma')
        else:
            forma=None
        if st.checkbox('dado flutuante'):
            hover_data = st.multiselect('', df_temp.columns)
        else:
            hover_data = None
        if st.button('Gerar gráfico'):
            st.plotly_chart(px.scatter(df_temp, x=x, y=y, color=color, size=size, symbol=forma, hover_data=hover_data), 
                                use_container_width=True)