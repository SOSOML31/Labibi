import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import heapq

# =======================
# Génération du labyrinthe avec Prim
# =======================
def generate_prim(w, h):
    maze = [[1] * w for _ in range(h)]  # Initialisation avec des murs partout
    start_x, start_y = random.randint(0, h - 1), random.randint(0, w - 1)
    maze[start_x][start_y] = 0  # Point de départ
    walls = [(start_x + dx, start_y + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)] if 0 <= start_x + dx < h and 0 <= start_y + dy < w]

    while walls:
        wx, wy = walls.pop(random.randint(0, len(walls) - 1))
        if maze[wx][wy] == 1:
            neighbors = [(wx + dx, wy + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)] if 0 <= wx + dx < h and 0 <= wy + dy < w and maze[wx + dx][wy + dy] == 0]
            if len(neighbors) == 1:
                maze[wx][wy] = 0
                walls.extend([(wx + dx, wy + dy) for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)] if 0 <= wx + dx < h and 0 <= wy + dy < w and maze[wx + dx][wy + dy] == 1])

    return maze

# =======================
# Algorithme A*
# =======================
def a_star(maze, start, end):
    h, w = len(maze), len(maze[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(end[0] - start[0]) + abs(end[1] - start[1])}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w and maze[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + abs(end[0] - neighbor[0]) + abs(end[1] - neighbor[1])
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

# =======================
# Génération des données
# =======================
def generate_dataset(num_mazes, w, h):
    data = []
    for _ in range(num_mazes):
        maze = generate_prim(w, h)
        free_cells = [(i, j) for i in range(h) for j in range(w) if maze[i][j] == 0]
        start, end = random.sample(free_cells, 2)
        path = a_star(maze, start, end)
        if path:
            data.append((maze, start, end, path))
    return data

# Sauvegarder le jeu de données
dataset = generate_dataset(1000, 10, 10)
with open('maze_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print("Jeu de données généré et sauvegardé dans 'maze_dataset.pkl'.")

# =======================
# Préparation des données
# =======================
def preprocess_data(dataset):
    X, y = [], []
    for maze, start, end, path in dataset:
        X.append(np.array(maze).flatten())
        target = np.zeros_like(maze, dtype=float)
        for px, py in path:
            target[px][py] = 1.0
        y.append(target.flatten())
    return np.array(X), np.array(y)

# Charger le jeu de données
with open('maze_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

X, y = preprocess_data(dataset)

# =======================
# Définition du modèle
# =======================
class MazeSolverNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MazeSolverNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialiser le modèle
input_size = 100  # 10x10 labyrinthe aplati
hidden_size = 128
output_size = 100  # 10x10 chemin aplati
model = MazeSolverNet(input_size, hidden_size, output_size)
print("Modèle de réseau de neurones initialisé.")

# =======================
# Entraînement du modèle
# =======================
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Entraînement terminé.")

# =======================
# Prédiction avec le modèle
# =======================
def predict_path(model, maze):
    maze_flat = np.array(maze).flatten()
    maze_tensor = torch.tensor(maze_flat, dtype=torch.float32).unsqueeze(0)
    output = model(maze_tensor).detach().numpy().reshape(maze.shape)
    predicted_path = np.where(output > 0.5, 1, 0)
    return predicted_path

# Tester la prédiction
test_maze = dataset[0][0]
predicted_path = predict_path(model, test_maze)
print("Labyrinthe :")
print(np.array(test_maze))
print("Chemin prédit :")
print(predicted_path)