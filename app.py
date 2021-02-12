import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout='wide', page_title='DataScience')
px.set_mapbox_access_token('pk.eyJ1Ijoicm9uYWxkb2xhZ2UiLCJhIjoiY2tldm5yMnl1MGNxcTJ6bm9qcnR6eDl0ZSJ9.SHArbKTHFkYTNFaVro_gAw')

@st.cache
def get_data(exemplos, option):
    return exemplos[option]


def main():
    

    df1 = pd.read_csv('dados/airbnb_ny2.csv')
    df2 = pd.read_csv('dados/alugueis_brasil.csv')

    exemplos = {'AirBnb Nova Iorque': df1, 
                'Alugueis em São Paulo': df2}

    
    st.title('Ferramentas Estatísticas Aplicadas')
    st.header('Análise exploratória de dados')
    option = st.selectbox('Escolha o exemplo', [key for key in exemplos])



    st.subheader('Dados:')
    df = get_data(exemplos, option)
    st.dataframe(df)

    with st.beta_expander('Estatísticas agrupadas'):
        todas_colunas = list(df.columns)
        groupy_list = st.multiselect('Escolha as colunas de agrupamento', todas_colunas)
        colunas_restantes = [value for value in todas_colunas if value not in groupy_list]
        target_list = st.multiselect('Escolha as colunas da estatística', colunas_restantes)
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
            df_temp = estatisticas[statistics]
            st.dataframe(df_temp)
            
    with st.beta_expander('Gráficos interativos'):
        lista_graficos = ['dispersão', 'linha', 'area', 'histograma', 'box', 'violino', 'densidade', 'categoria paralela',
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

if __name__ == '__main__':
    user = st.sidebar.text_input('Usuário')
    password = st.sidebar.text_input('Senha de acesso', type='password')
    
    if st.sidebar.button('Entrar'):
        if user == 'alunos' and password == 'unifor':
            main()
    elif user == 'alunos' and password == 'unifor':
        main()       

