from PIL import Image
import random
import heapq

# =======================
# Affichage du labyrinthe
# =======================
def print_maze(lab, in_out=None, path=None):
    """
    Affiche le labyrinthe dans la console.
    'i' = entrée, 'o' = sortie, '█' = mur, '░' = passage libre, '.' = chemin.
    """
    for i, row in enumerate(lab):
        line = ""
        for j, cell in enumerate(row):
            if in_out and (i, j) == in_out[0]:  # Entrée
                line += "\033[92mi\033[0m"  # Vert pour l'entrée
            elif in_out and (i, j) == in_out[1]:  # Sortie
                line += "\033[91mo\033[0m"  # Rouge pour la sortie
            elif path and (i, j) in path:  # Chemin
                line += "\033[94m.\033[0m"  # Bleu pour le chemin
            else:
                line += "░" if cell == 0 else "█"
        print(line)

# =======================
# Génération d'un labyrinthe avec Prim
# =======================
def generate_prim(w, h):
    """
    Génère un labyrinthe en utilisant l'algorithme de Prim.
    """
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
# A* Algorithm
# =======================
def a_star(maze, start, end):
    """
    Implémente l'algorithme A* pour trouver le chemin le plus court.
    """
    h, w = len(maze), len(maze[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(end[0] - start[0]) + abs(end[1] - start[1])}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            # Reconstruire le chemin
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
# Dijkstra Algorithm
# =======================
def dijkstra(maze, start, end):
    """
    Implémente l'algorithme de Dijkstra pour trouver le chemin le plus court.
    """
    h, w = len(maze), len(maze[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        current_g, current = heapq.heappop(open_set)
        if current == end:
            # Reconstruire le chemin
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w and maze[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = current_g + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    heapq.heappush(open_set, (tentative_g_score, neighbor))
    return []

# =======================
# Comparaison entre A* et Dijkstra
# =======================
def compare_algorithms(maze, start, end):
    """
    Compare les chemins trouvés par A* et Dijkstra.
    """
    path_a_star = a_star(maze, start, end)
    path_dijkstra = dijkstra(maze, start, end)

    if path_a_star == path_dijkstra:
        print("\nLes chemins trouvés par A* et Dijkstra sont identiques.")
    else:
        print("\nLes chemins trouvés par A* et Dijkstra sont différents.")
        print("Chemin A* :", path_a_star)
        print("Chemin Dijkstra :", path_dijkstra)

# =======================
# Génération d'une image PNG
# =======================
def fancy_maze(bin_lab_2d, name, in_out=None, path=None):
    """
    Génère une image PNG du labyrinthe avec l'entrée (vert), la sortie (rouge), et le chemin (bleu).
    """
    h, w = len(bin_lab_2d), len(bin_lab_2d[0])
    img = Image.new("RGB", (w, h), "black")
    pixels = img.load()
    for i in range(h):
        for j in range(w):
            if in_out and (i, j) == in_out[0]:  # Entrée en vert
                pixels[j, i] = (0, 255, 0)
            elif in_out and (i, j) == in_out[1]:  # Sortie en rouge
                pixels[j, i] = (255, 0, 0)
            elif path and (i, j) in path:  # Chemin en bleu
                pixels[j, i] = (0, 0, 255)
            else:
                pixels[j, i] = (255, 255, 255) if bin_lab_2d[i][j] == 0 else (0, 0, 0)
    img.save(f"{name}.png")

# =======================
# Test complet
# =======================
def test_maze_with_path():
    """
    Test pour générer un labyrinthe et comparer les chemins avec A* et Dijkstra.
    """
    w, h = 20, 20
    maze = generate_prim(w, h)
    
    # Choisir l'entrée et la sortie
    free_cells = [(i, j) for i in range(h) for j in range(w) if maze[i][j] == 0]
    start, end = random.sample(free_cells, 2)
    
    # Trouver le chemin avec A*
    path_a_star = a_star(maze, start, end)
    print("\nLabyrinthe avec chemin A* :")
    print_maze(maze, (start, end), path_a_star)
    fancy_maze(maze, "maze_with_a_star", (start, end), path_a_star)
    
    # Comparer avec Dijkstra
    compare_algorithms(maze, start, end)

# Lancer le test
if __name__ == "__main__":
    test_maze_with_path()