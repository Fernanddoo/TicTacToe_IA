import pygame
import sys
import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np
import pickle
import os
import random

pygame.init()

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Configurações da tela
SIZE = WIDTH, HEIGHT = 480, 480
LINE_WIDTH = 5
SCREEN = pygame.display.set_mode(SIZE)
pygame.display.set_caption('Jogo da Velha com IA')

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 9)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Iniciar rede neural e otimizador
model = TicTacToeNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

data_file = 'training_data.pkl'
model_file = 'tic_tac_toe_model.pth'

# Carregar dados de treinamento existentes
if os.path.exists(data_file):
    with open(data_file, 'rb') as f:
        training_data = pickle.load(f)
else:
    training_data = []

# Carregar o modelo treinado, se existir
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file))
    model.eval()

# Treinar o modelo com dados acumulados
def train_model(training_data):
    if not training_data:
        return  # Sem dados para treinar

    model.train()
    batch_size = 32  # Fine tuning
    num_epochs = 5  

    # Preparar dados para treinamento
    states = []
    targets = []
    for state, action, reward in training_data:
        state_tensor = torch.tensor(state, dtype=torch.float)
        target = model(state_tensor.unsqueeze(0)).detach().squeeze(0)
        target[action] = reward
        states.append(state_tensor)
        targets.append(target)

    # Criar dataset e dataloader
    dataset = torch.utils.data.TensorDataset(torch.stack(states), torch.stack(targets))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loop de treinamento
    for epoch in range(num_epochs):
        for batch_states, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

# Treinar o modelo com os dados carregados
train_model(training_data)
torch.save(model.state_dict(), model_file)

def draw_board():
    SCREEN.fill(WHITE)
    # Linhas verticais
    pygame.draw.line(SCREEN, BLACK, (WIDTH / 3, 0), (WIDTH / 3, HEIGHT), LINE_WIDTH)
    pygame.draw.line(SCREEN, BLACK, (2 * WIDTH / 3, 0), (2 * WIDTH / 3, HEIGHT), LINE_WIDTH)
    # Linhas horizontais
    pygame.draw.line(SCREEN, BLACK, (0, HEIGHT / 3), (WIDTH, HEIGHT / 3), LINE_WIDTH)
    pygame.draw.line(SCREEN, BLACK, (0, 2 * HEIGHT / 3), (WIDTH, 2 * HEIGHT / 3), LINE_WIDTH)

def draw_markers(board):
    for row in range(3):
        for col in range(3):
            if board[row][col] == 1:
                # Desenhar X
                pygame.draw.line(SCREEN, BLACK, (col * WIDTH / 3 + 20, row * HEIGHT / 3 + 20),
                                 ((col + 1) * WIDTH / 3 - 20, (row + 1) * HEIGHT / 3 - 20), LINE_WIDTH)
                pygame.draw.line(SCREEN, BLACK, ((col + 1) * WIDTH / 3 - 20, row * HEIGHT / 3 + 20),
                                 (col * WIDTH / 3 + 20, (row + 1) * HEIGHT / 3 - 20), LINE_WIDTH)
            elif board[row][col] == -1:
                # Desenhar O
                pygame.draw.circle(SCREEN, BLACK, (int(col * WIDTH / 3 + WIDTH / 6), int(row * HEIGHT / 3 + HEIGHT / 6)),
                                   WIDTH // 6 - 20, LINE_WIDTH)

# Função para verificar se um jogador venceu
def check_win(board, player):
    # Verificar linhas, colunas e diagonais
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    if board[0, 0] == board[1, 1] == board[2, 2] == player or board[0, 2] == board[1, 1] == board[2, 0] == player:
        return True
    return False

# Função para verificar empate
def check_draw(board):
    return np.all(board != 0)

def board_to_tensor(board):
    return torch.tensor(board.flatten(), dtype=torch.float).unsqueeze(0)

# Função para a IA escolher o melhor movimento
def ai_move(board, epsilon=0.1):
    # Obter movimentos válidos
    valid_moves = np.argwhere(board.flatten() == 0)
    if len(valid_moves) == 0:
        return None
    
    if random.random() < epsilon:
        #Tenta ser mais exploradora
        move = random.choice(valid_moves)[0]
    else:
        # Rede neural faz a predição
        with torch.no_grad():
            output = model(board_to_tensor(board))
        # Aplicar máscara para movimentos inválidos
        output_numpy = output.numpy().flatten()
        output_numpy[board.flatten() != 0] = -np.inf
        # Escolher a ação com maior valor
        move = np.argmax(output_numpy)

    row, col = divmod(move, 3)
    return row, col

# Variáveis do jogo
board = np.zeros((3, 3), dtype=int)
player = 1  # Jogador humano começa
game_over = False
data = []  # Armazenar estados e ações para treinamento
winner = None

# Loop principal do jogo
while True:
    draw_board()
    draw_markers(board)
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if not game_over:
            if player == 1:
                # Movimento do jogador humano
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    row = int(y // (HEIGHT / 3))
                    col = int(x // (WIDTH / 3))
                    if board[row][col] == 0:
                        board[row][col] = player
                        if check_win(board, player):
                            game_over = True
                            winner = 'Você venceu!'
                            if len(data) > 0:
                                # A última entrada da IA foi antes dessa jogada humana
                                # Recompensa do último movimento da IA para -1, aprender com a derrota
                                last_state, last_action, _ = data[-1]
                                data[-1] = (last_state, last_action, -1)
                        elif check_draw(board):
                            game_over = True
                            winner = 'Deu velha!'
                            # Empate: última jogada da IA não é ruim nem boa
                            if len(data) > 0:
                                last_state, last_action, _ = data[-1]
                                data[-1] = (last_state, last_action, 0.5)
                        else:
                            player = -1  # Alternar para a IA
            else:
                # Movimento da IA
                row_col = ai_move(board)
                if row_col:
                    row, col = row_col
                    board[row][col] = player
                    # Armazenar estado e ação APENAS para a IA
                    state = board.copy()
                    action = row * 3 + col
                    data.append((state.flatten(), action, 0))  # IA move com recompensa inicial 0
                    if check_win(board, player):
                        game_over = True
                        winner = 'A IA venceu!'
                        # Vitória da IA
                        last_state, last_action, _ = data[-1]
                        data[-1] = (last_state, last_action, 1)
                    elif check_draw(board):
                        game_over = True
                        winner = 'Deu velha!'
                        last_state, last_action, _ = data[-1]
                        data[-1] = (last_state, last_action, 0.5)
                    else:
                        player = 1  # Alternar para o jogador humano
                else:
                    # Se a IA não encontrou movimento válido, é empate
                    game_over = True
                    winner = 'Deu velha!'

    if game_over:
        # Exibir mensagem de fim de jogo
        font = pygame.font.SysFont(None, 40)
        text = font.render(winner, True, BLACK)
        text_rect = text.get_rect(center=(WIDTH / 2, HEIGHT / 2))
        SCREEN.blit(text, text_rect)
        pygame.display.update()
        pygame.time.wait(2000)

        train_model(data)
        torch.save(model.state_dict(), model_file)

        # Salvar os dados acumulados
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                existing_data = pickle.load(f)
        else:
            existing_data = []

        existing_data.extend(data)

        with open(data_file, 'wb') as f:
            pickle.dump(existing_data, f)

        # Resetar o jogo
        board = np.zeros((3, 3), dtype=int)
        player = 1
        game_over = False
        data = []
        winner = None
