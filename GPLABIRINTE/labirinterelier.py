from PIL import Image
import random




# =======================
# Affichage du labyrinthe

def print_maze(lab, in_out=None):
    """
    Affiche le labyrinthe dans la console avec l'entrée ('i') et la sortie ('o').
    """
    for i, row in enumerate(lab):
        line = ""
        for j, cell in enumerate(row):
            if in_out and (i, j) == in_out[0]:  # Entrée
                line += "\033[92mi\033[0m"  # Vert pour l'entrée
            elif in_out and (i, j) == in_out[1]:  # Sortie
                line += "\033[91mo\033[0m"  # Rouge pour la sortie
            else:
                line += "░" if cell == 0 else "█"
        print(line)



# =======================
# Génération d'un labyrinthe avec Prim
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
# Vérification d'un chemin entre l'entrée et la sortie

def has_path(maze, start, end):
    """
    Vérifie s'il existe un chemin entre l'entrée et la sortie avec DFS.
    """
    h, w = len(maze), len(maze[0])
    stack = [start]
    visited = set()

    while stack:
        x, y = stack.pop()
        if (x, y) == end:
            return True
        if (x, y) in visited:
            continue
        visited.add((x, y))
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and maze[nx][ny] == 0 and (nx, ny) not in visited:
                stack.append((nx, ny))
    return False

# =======================
# Générer un labyrinthe avec un chemin relié


def generate_maze_with_path(w, h):
    """
    Génère un labyrinthe avec un chemin garanti entre l'entrée et la sortie sur les bords.
    """
    while True:
        maze = generate_prim(w, h)
        
        # Choisir des points d'entrée et de sortie sur les bords
        free_cells = [(0, j) for j in range(w) if maze[0][j] == 0] + \
                     [(h-1, j) for j in range(w) if maze[h-1][j] == 0] + \
                     [(i, 0) for i in range(h) if maze[i][0] == 0] + \
                     [(i, w-1) for i in range(h) if maze[i][w-1] == 0]
        
        if len(free_cells) >= 2:
            start, end = random.sample(free_cells, 2)
            if has_path(maze, start, end):
                return maze, (start, end)




# =======================
# Génération d'une image PNG
# =======================
def fancy_maze(bin_lab_2d, name, in_out=None):
    """
    Génère une image PNG du labyrinthe avec entrée (vert) et sortie (rouge).
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
            else:
                pixels[j, i] = (255, 255, 255) if bin_lab_2d[i][j] == 0 else (0, 0, 0)
    img.save(f"{name}.png")



# =======================
# Test complet
# =======================
def test_maze_with_path():
    """
    Test pour générer un labyrinthe avec un chemin entre l'entrée et la sortie sur les bords.
    """
    w, h = 20, 20  # Taille du labyrinthe
    maze, in_out = generate_maze_with_path(w, h)
    print("\nLabyrinthe avec chemin garanti :")
    print_maze(maze, in_out)
    fancy_maze(maze, "maze_with_path", in_out)
    print(f"\nImage sauvegardée sous 'maze_with_path.png'.\nEntrée : {in_out[0]}, Sortie : {in_out[1]}")

# Lancer le test
if __name__ == "__main__":
    test_maze_with_path()