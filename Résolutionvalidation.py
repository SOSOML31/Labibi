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
    return []  # Retourne une liste vide si aucun chemin n'est trouvé

# =======================
# Vérification de la validité d'un chemin
# =======================
def check_path(maze, path):
    """
    Vérifie que le chemin donné est valide :
    - Chaque case du chemin est adjacente à la précédente.
    - Toutes les cases du chemin sont des passages libres.
    """
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        # Vérifie que les cases sont adjacentes
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return False
        # Vérifie que les cases du chemin sont libres
        if maze[x1][y1] != 0 or maze[x2][y2] != 0:
            return False
    return True

# =======================
# Génération d'une image "real_fancy"
# =======================
def real_fancy_maze(maze, name, in_out, path):
    """
    Génère une image PNG avec l'entrée (indigo), la sortie (orange) et le chemin (violet).
    """
    h, w = len(maze), len(maze[0])
    img = Image.new("RGB", (w, h), "black")
    pixels = img.load()
    for i in range(h):
        for j in range(w):
            if (i, j) == in_out[0]:  # Entrée en indigo
                pixels[j, i] = (75, 0, 130)
            elif (i, j) == in_out[1]:  # Sortie en orange
                pixels[j, i] = (255, 165, 0)
            elif path and (i, j) in path:  # Chemin en violet
                pixels[j, i] = (128, 0, 128)
            else:
                pixels[j, i] = (255, 255, 255) if maze[i][j] == 0 else (0, 0, 0)
    img.save(f"{name}.png")

# =======================
# Test complet
# =======================
def test_real_fancy_maze():
    """
    Test complet avec génération, résolution et validation.
    """
    w, h = 20, 20  # Taille du labyrinthe
    maze = generate_prim(w, h)  # Générer le labyrinthe
    
    # Obtenir des cellules libres pour start et end
    free_cells = [(i, j) for i in range(h) for j in range(w) if maze[i][j] == 0]
    start, end = random.sample(free_cells, 2)
    
    # Vérifications
    assert maze[start[0]][start[1]] == 0, f"L'entrée {start} est sur un mur !"
    assert maze[end[0]][end[1]] == 0, f"La sortie {end} est sur un mur !"
    print(f"Entrée : {start}, Sortie : {end}")
    
    # Trouver un chemin avec A*
    path = a_star(maze, start, end)
    if not path:
        print("Aucun chemin trouvé avec A*.")
        return
    
    # Vérifier le chemin
    assert check_path(maze, path), "Le chemin trouvé n'est pas valide !"
    
    # Afficher le labyrinthe avec le chemin
    print("\nLabyrinthe avec chemin (A*):")
    print_maze(maze, (start, end), path)
    
    # Générer une image "real_fancy"
    real_fancy_maze(maze, "real_fancy_maze", (start, end), path)
    print("\nImage 'real_fancy_maze.png' générée avec succès.")

# Lancer le test
if __name__ == "__main__":
    test_real_fancy_maze()