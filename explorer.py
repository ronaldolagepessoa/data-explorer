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

def explorer():
    st.header('Análise exploratória de dados')
    exemplos = { 
                'AirBnb Nova Iorque': {'data': 'airbnb_ny2', 'sep': ','},
                'Alugueis em São Paulo': {'data': 'alugueis_brasil', 'sep': ','},
                'Alugueis no Brasil': {'data': 'alugueis_brasil2', 'sep': ','},
                'Banco': {'data': 'bank-additional-full', 'sep': ';'}}
    option = st.selectbox('Escolha o exemplo', [key for key in exemplos])
    with st.beta_expander('Dados'):
        linhas = st.slider('Número de linhas para exibir', value=5, max_value=100, min_value=5)
        df_temp = fecth_data(exemplos[option]['data'], sep=exemplos[option]['sep'])
        st.dataframe(df_temp.head(linhas))
        st.write(f'Número total de linhas = {df_temp.shape[0]}')
    with st.beta_expander('Descrição'):
        if option == 'AirBnb Nova Iorque':
            with open('markdowns/airbnb_ny', 'r') as file:
                st.markdown(file.read())
        elif option == 'Alugueis em São Paulo':
            with open('markdowns/alugueis_brasil', 'r') as file:
                st.markdown(file.read())
        elif option == 'Alugueis no Brasil':
            with open('markdowns/alugueis_brasil2', 'r') as file:
                st.markdown(file.read())
        elif option == 'Banco':
            with open('markdowns/banco', 'r') as file:
                st.markdown(file.read())
            

    with st.beta_expander('Filtros'):
        filtro = st.empty()
        cols = st.beta_columns(2)
        st.subheader('Colunas categóricas')
        with cols[0]:
            st.subheader('Colunas Categóricas')
            freqs = {}
            for column in df_temp.select_dtypes(include='object').columns:
                st.write(column)
                freqs[column, 'mínimo'] = st.number_input(f"Frequência mínima", 
                                                    min_value=df_temp[column].value_counts().min(), 
                                                    max_value=df_temp[column].value_counts().max(),
                                                    value=df_temp[column].value_counts().min(), key=column)
                freqs[column, 'máximo'] = st.number_input(f"Frequência máxima", 
                                                    min_value=df_temp[column].value_counts().min(), 
                                                    max_value=df_temp[column].value_counts().max(),
                                                    value=df_temp[column].value_counts().max(), key=column)
        with cols[1]:
            st.subheader('Colunas Numéricas')
            values = {}
            for column in df_temp.select_dtypes(exclude='object').columns:
                st.write(column)
                values[column, 'mínimo'] = st.number_input(f"Valor mínimo", value=df_temp[column].min(), key=column)
                values[column, 'máximo'] = st.number_input(f"Valor máximo", value=df_temp[column].max(), key=column)
        if filtro.checkbox('Aplicar filtros'):
            df = df_temp.copy()
            for column in df_temp.select_dtypes(include='object').columns:
                lista = df_temp[column].value_counts()
                v = lista[lista.values >= freqs[column, 'mínimo']].index
                filtro = df[column].isin(v)
                df = df.loc[filtro].copy()
                v = lista[lista <= freqs[column, 'máximo']].index
                filtro = df[column].isin(v)
                df = df.loc[filtro].copy()
            for column in df_temp.select_dtypes(exclude='object').columns:
                filtro = (df_temp[column].between(values[column, 'mínimo'], values[column, 'máximo']))
                df = df.loc[filtro]
        else:
            df = df_temp.copy()

    with st.beta_expander('Estatísticas por coluna'):
        todas_colunas = list(df.columns)
        target_column = st.selectbox('Escolha a coluna para cálculo da estatística', todas_colunas)
        if target_column in df.select_dtypes(exclude='object').columns:
            lista_estatisticas = ['média', 
                        'desvio padrão', 
                        'mediana', 
                        'mínimo', 
                        'máximo', 
                        'estatísticas descritivas básicas']
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            statistics = st.radio('Escolha a estatística', lista_estatisticas, key='por coluna')
            if st.button('Calcular', key='por coluna'):
                estatisticas = {'média': df[target_column].mean(), 
                    'desvio padrão': df[target_column].std(), 
                    'mediana': df[target_column].median(), 
                    'mínimo': df[target_column].min(), 
                    'máximo': df[target_column].max(), 
                    'estatísticas descritivas básicas': df[target_column].describe().rename(index={'count': 'freq',
                                                                                                   'mean': 'média', 
                                                                                                   'std': 'desvpad'})}
                df_e = estatisticas[statistics]
                try:
                    st.dataframe(df_e)
                except:
                    st.write(df_e)
        else:
            lista_estatisticas = ['valores únicos', 
                        'freq valores únicos', 
                        'freq valores únicos (normalizado)',
                        'estatísticas descritivas básicas']
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            statistics = st.radio('Escolha a estatística', lista_estatisticas, key='por coluna')
            if st.button('Calcular', key='por coluna'):
                estatisticas = {'valores únicos': df[target_column].unique(), 
                    'freq valores únicos': df[target_column].value_counts(), 
                    'freq valores únicos (normalizado)': df[target_column].value_counts(normalize=True), 
                    'estatísticas descritivas básicas': df[target_column].describe()}
                df_e = estatisticas[statistics]
                try:
                    st.dataframe(df_e)
                except:
                    st.write(df_e)

    with st.beta_expander('Estatísticas agrupadas'):
        todas_colunas = list(df.columns)
        groupy_list = st.multiselect('Escolha as colunas de agrupamento', todas_colunas)
        # colunas_restantes = [value for value in todas_colunas if value not in groupy_list]
        target_list = st.multiselect('Escolha as colunas da estatística', todas_colunas)
        lista_estatisticas = ['média', 
                        'desvio padrão', 
                        'mediana', 
                        'mínimo', 
                        'máximo', 
                        'estatísticas descritivas básicas']
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        statistics = st.radio('Escolha a estatística', lista_estatisticas) 
        if st.button('Calcular'):
            estatisticas = {'média': df.groupby(groupy_list)[target_list].mean(), 
                        'desvio padrão': df.groupby(groupy_list)[target_list].std(), 
                        'mediana': df.groupby(groupy_list)[target_list].median(), 
                        'mínimo': df.groupby(groupy_list)[target_list].min(), 
                        'máximo': df.groupby(groupy_list)[target_list].max(), 
                        'estatísticas descritivas básicas': df.groupby(groupy_list)[target_list].describe().rename(columns={'count': 'freq',
                                                                                                                        'mean': 'média', 
                                                                                                                        'std': 'desvpad'
                                                                                                                        })}
            df_e = estatisticas[statistics]
            st.dataframe(df_e.reset_index())
            
    with st.beta_expander('Gráficos interativos'):
        lista_graficos = ['dispersão','densidade', 'linha', 'area', 'histograma', 'box', 'violino', 'categoria paralela',
                        'dispersão geográfica', 'densidade geográfica']
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        tipo_grafico = st.radio('Tipo de gráfico', lista_graficos)
        if tipo_grafico == 'dispersão': 
            x = st.selectbox('x', df.columns)
            y = st.selectbox('y', df.columns)
            if st.checkbox('cor'):
                color = st.selectbox('', df.columns, key='color')
            else:
                color=None
            if st.checkbox('tamanho'):
                size = st.selectbox('', df.columns, key='size')
            else:
                size=None
            if st.checkbox('dado flutuante'):
                hover_data = st.multiselect('', df.columns)
            else:
                hover_data = None
            if st.button('Gerar gráfico'):
                st.plotly_chart(px.scatter(df, x=x, y=y, color=color, size=size, hover_data=hover_data), 
                                use_container_width=True)
        elif tipo_grafico == 'linha': 
            x = st.selectbox('x', df.columns)
            y = st.selectbox('y', df.columns)
            if st.checkbox('cor'):
                color = st.selectbox('', df.columns)
            else:
                color=None
            if st.checkbox('dado flutuante'):
                hover_data = st.multiselect('', df.columns)
            else:
                hover_data = None
            if st.button('Gerar gráfico'):
                st.plotly_chart(px.line(df, x=x, y=y, color=color, hover_data=hover_data), 
                                use_container_width=True)
        elif tipo_grafico == 'area': 
            x = st.selectbox('x', df.columns)
            y = st.selectbox('y', df.columns)
            if st.checkbox('cor'):
                color = st.selectbox('', df.columns)
            else:
                color=None
            if st.checkbox('dado flutuante'):
                hover_data = st.multiselect('', df.columns)
            else:
                hover_data = None
            if st.button('Gerar gráfico'):
                st.plotly_chart(px.area(df, x=x, y=y, color=color, hover_data=hover_data), 
                                use_container_width=True)
        elif tipo_grafico == 'histograma': 
            x = st.selectbox('x', df.columns)
            if st.checkbox('cor'):
                color = st.selectbox('', df.columns)
            else:
                color=None
            if st.button('Gerar gráfico'):
                st.plotly_chart(px.histogram(df, x=x, color=color), 
                                use_container_width=True)
        elif tipo_grafico == 'box': 
            x = st.selectbox('x', df.columns)
            y = st.selectbox('y', df.columns)
            if st.checkbox('cor'):
                color = st.selectbox('', df.columns)
            else:
                color=None
            if st.button('Gerar gráfico'):
                st.plotly_chart(px.box(df, x=x, y=y, color=color), 
                                use_container_width=True)
        elif tipo_grafico == 'violino': 
            x = st.selectbox('x', df.columns)
            y = st.selectbox('y', df.columns)
            if st.checkbox('cor'):
                color = st.selectbox('', df.columns)
            else:
                color=None
            if st.checkbox('mostrar pontos'):
                points='all'
            else:
                points=False
            if st.button('Gerar gráfico'):
                st.plotly_chart(px.violin(df, x=x, y=y, color=color, points=points), 
                                use_container_width=True)
        elif tipo_grafico == 'densidade': 
            x = st.selectbox('x', df.columns)
            y = st.selectbox('y', df.columns)
            tipo = st.radio('tipo', ['mapa de cor', 'curva de densidade'])
            if st.button('Gerar gráfico'):
                if tipo == 'mapa de cor':
                    st.plotly_chart(px.density_heatmap(df, x=x, y=y), 
                                use_container_width=True)
                else:
                    st.plotly_chart(px.density_contour(df, x=x, y=y), 
                                use_container_width=True)
        elif tipo_grafico == 'categoria paralela': 
            dimensions = st.multiselect('dimensões', df.columns)
            if st.button('Gerar gráfico'):
                st.plotly_chart(px.parallel_categories(df, dimensions=dimensions), 
                            use_container_width=True)
        elif tipo_grafico == 'dispersão geográfica': 
            lat = st.selectbox('latitude', df.columns)
            lon = st.selectbox('longitude', df.columns)
            if st.checkbox('cor'):
                color = st.selectbox('', df.columns)
            else:
                color=None
            if st.checkbox('tamanho'):
                size = st.selectbox('', df.columns, key='size map')
            else:
                size=None
            if st.checkbox('dado flutuante'):
                hover_data = st.multiselect('', df.columns, key='hover map')
            else:
                hover_data = None
            if st.button('Gerar gráfico'):
                st.plotly_chart(px.scatter_mapbox(df, lat=lat, lon=lon, color=color, size=size, hover_data=hover_data), 
                                use_container_width=True)
        elif tipo_grafico == 'densidade geográfica': 
            lat = st.selectbox('latitude', df.columns)
            lon = st.selectbox('longitude', df.columns)
            if st.button('Gerar gráfico'):
                st.plotly_chart(px.density_mapbox(df, lat=lat, lon=lon), 
                                use_container_width=True)