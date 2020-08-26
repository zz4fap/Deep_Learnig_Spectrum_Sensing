# Deep Learning for Spectrum Sensing

Código utilizado gerar, treinar e avaliar os modelos propostos no artigo.

## Dataset

Os datasets utilizados possuem tamanho em torno de 15 GB. Devido ao tamanho não foi possível armazena-los no repositório. 

Os scripts para gerar os datasets estão disponíveis na pasta **_generate_dataset** . Dependências necessárias :

- python 2.7;
- Software GNU Radio com as devidas variáveis de ambiente apontadas.

## Ambiente

Para rodar os scripts de treinamento e validação foram utilizados:

- Python 3.8
- Tensorflow 2.2
- Numpy
- Matplotlib
- Pandas
- Sklearn
- Pickle

## Observação

Há uma incompatibilidade ao rodar o modelo DetectNet utilizando o tensorflow CPU devido a camada de convolução utilizada. Para contornar esse problema utiliza-se da flag **swap_dim = True** para trocar duas dimensões do dataset de lugar e assim ser possível executar o script.  
