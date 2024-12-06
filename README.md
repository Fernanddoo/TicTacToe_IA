# Trabalho de IA

Como instalar:

(Recomendável criar em um ambiente virtual para gerenciar as dependências do seu projeto sem afetar outros projetos em seu sistema.)

Clone ou baixe o arquivo tictactoe.py e execute os seguintes comandos no terminal:

- pip install torch torchvision
- pip install pygame
- pip install numpy

Após a instalação, use o comando:
- python3 .\tictactoe.py
Ou aperte run, caso use o vscode

# Documentação

Bibliotecas utilizadas:

- Pygame:
    - O pygame é uma biblioteca para a criação de jogos 2D em Python.
    - Uso no código: Inicialização da janela do jogo, Desenho do tabuleiro, Captura de eventos do mouse e atualiza a cada iteração do loop principal do jogo

- Torch (PyTorch):
    - O Torch é uma biblioteca de aprendizado de máquina open-source
    - Uso no código: Definição de uma classe de rede neural (TicTacToeNet) herdando nn.Module, Uso de funções de ativação (ex: torch.relu) para não deixar o modelo com linearidade, uma função de perda (nn.MSELoss) para calcular a recompensa/ação tomada do modelo, Criação de datasets e dataloaders para iterar sobre dados de treinamento armazenados e possui persistência do estado do modelo treinado.

- Numpy:
    - biblioteca fundamental para computação científica em Python
    - Uso no código: Representação do tabuleiro do jogo da velha como uma matriz numpy 3x3, manipulações de arrays para verificar vitórias, empates e para transformar o estado do tabuleiro em vetores para entrada no modelo conversão entre arrays numpy e tensores do torch.

- Random:
    - biblioteca nativa do python, fornece funções para geração de números aleatórios.
    - Uso no código: Escolha de movimentos aleatórios pela IA com certa probabilidade (epsilon) durante a tomada de decisão, garantindo um equilíbrio entre exploração e exploração da política aprendida pela rede neural.

# Escopo do trabalho:

Ao integrar essas tecnologias, o código cria um loop de jogo interativo (via pygame), registra estados/ações/recompensas, e utiliza esses dados para treinar uma rede neural (torch), armazenando o histórico (via pickle) para aprendizado contínuo.

Com o uso de todas essas bibliotecas e tecnologias, permite que o modelo aprenda com cada partida jogada, tornando-se um jogador cada vez mais experiente em TicTacToe (Jogo da Velha).

Foi selecionado treinamento gradual, para poder ser observado de perto, a cada loop de partida, a melhora continua nas estratégias do modelo.
