import streamlit as st
from PIL import Image
from pathlib import Path
import base64
from explorer import explorer
from correlation import correlation
from regression import regression
from autoregression import autoregression


st.set_page_config(layout='wide', page_title='DataScience')





def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def main():
    st.title('Ferramentas Estatísticas Aplicadas')
    
    with st.beta_expander('Clique para exibir informações sobre o autor da ferramenta'):
        st.markdown(r"""### Sobre o autor""")
        cols = st.beta_columns((1, 8))
        with cols[0]:
            # img = Image.open('foto.jpg').convert('RGB')
            header_html = "<img src='data:image/png;base64,{}' class='img-fluid' width='100%'>".format(
            img_to_bytes("foto.jpg")
            )
            st.markdown(
                header_html, unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(r"""
                        #### Ronaldo Lage Pessoa
                        
                        ##### Formação
                        * Graduado em Engenharia de Produção Mecânica (UFC)
                        * Especialista em Gerenciamento de Projetos (FGV)
                        * Mestre em Modelagem e Métodos Quantitativos (UFC)
                        
                        ##### Atuação profissional
                        * Professor do curso do curso de Graduação em Administração (Unifor)
                        * Professor do curso de Pós-Graduação em Excelência Operacional (Unifor)
                        * Cientista de Dados (Profectum Tecnologia)
                        
                        ##### Contato
                        * ronaldo.lage.pessoa@gmail.com
                        """)
            
    opcao = st.sidebar.radio('Escolha uma opção', ('Análise exploratória de dados', 'Análise de correlação', 'Regressão', 'Séries Temporais'))
    if opcao == 'Análise exploratória de dados':
        explorer()
    elif opcao == 'Análise de correlação':
        correlation()
    elif opcao == 'Regressão':
        regression()
    elif opcao == 'Séries Temporais':
        autoregression()
    

if __name__ == '__main__':
    main()       

