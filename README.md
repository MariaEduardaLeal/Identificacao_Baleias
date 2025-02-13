 # Identificação de Baleias por Imagem

Este é um projeto de **classificação de imagens** que utiliza **redes neurais convolucionais (CNNs)** para identificar se duas imagens de cauda de baleia pertencem ao mesmo animal ou a indivíduos diferentes.

## 🚀 Tecnologias Utilizadas

- **Python** (para desenvolvimento geral)
- **TensorFlow/Keras** (para a CNN)
- **OpenCV** (para manipulação de imagens)
- **Tkinter** (para interface gráfica)

## 📌 Funcionalidades

- Carregamento de imagens para identificação
- Processamento e normalização das imagens
- Treinamento de um modelo de CNN
- Identificação de baleias baseado nas imagens carregadas
- Interface gráfica para interação com o usuário

## 🔧 Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
   cd nome-do-repositorio
   ```
2. Instale as dependências necessárias:
   ```bash
   pip install tensorflow opencv-python numpy pillow
   ```
3. Execute o programa:
   ```bash
   python app.py
   ```

## 📷 Exemplo de Uso

1. Clique em **"Carregar Imagem"** e selecione uma imagem de cauda de baleia.
2. O programa processará a imagem e a exibirá na interface.
3. Caso o modelo esteja treinado, a previsão será exibida na tela.
4. Se desejar, clique em **"Treinar Modelo"** para refinar as previsões.

## 🔜 Melhorias Futuras

- Usar um **dataset real** de caudas de baleias.
- Melhorar a arquitetura da CNN para obter maior precisão.
- Implementar uma **API Flask** para enviar imagens via web.

