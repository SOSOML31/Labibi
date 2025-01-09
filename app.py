from PIL import Image
import random
import heapq


# Conversion binary_map → graph_map
def bin_to_graph(binary_map, w):
    h = len(binary_map) // w
    graph_map = []
    for i in range(h):
        row = []
        for j in range(w):
            cell = [1, 1, 1, 1]
            if binary_map[i * w + j] == 0:
                if i > 0 and binary_map[(i - 1) * w + j] == 0:
                    cell[0] = 0
                if j < w - 1 and binary_map[i * w + (j + 1)] == 0:
                    cell[1] = 0
                if i < h - 1 and binary_map[(i + 1) * w + j] == 0:
                    cell[2] = 0
                if j > 0 and binary_map[i * w + (j - 1)] == 0:
                    cell[3] = 0
            row.append(cell)
        graph_map.append(row)
    return graph_map


# Conversion graph_map → binary_map
def graph_to_bin(graph_map):
    binary_map = []
    for row in graph_map:
        for cell in row:
            binary_map.append(0 if sum(cell) < 4 else 1)
    return binary_map


# Conversion flat → 2D
def flat_to_2d(flat_lab, w):
    return [flat_lab[i:i + w] for i in range(0, len(flat_lab), w)]


# Conversion 2D → flat
def flat_it(lab_2d):
    return [cell for row in lab_2d for cell in row]


# Impression du labyrinthe
def print_maze(lab, in_out=None, path=None):
    for i, row in enumerate(lab):
        line = ""
        for j, cell in enumerate(row):
            if in_out and (i, j) == in_out[0]:
                line += "i"
            elif in_out and (i, j) == in_out[1]:
                line += "o"
            elif path and (i, j) in path:
                line += "."
            else:
                line += "░" if cell == 0 else "█"
        print(line)


# Génération d'un labyrinthe avec l'algorithme de Prim
def generate_prim(w, h):
    maze = [[1] * w for _ in range(h)]
    walls = []
    start_x, start_y = random.randint(0, h - 1), random.randint(0, w - 1)
    maze[start_x][start_y] = 0
    walls.extend([(start_x + dx, start_y + dy, start_x, start_y)
                  for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                  if 0 <= start_x + dx < h and 0 <= start_y + dy < w])
    
    while walls:
        wx, wy, px, py = walls.pop(random.randint(0, len(walls) - 1))
        if 0 <= wx < h and 0 <= wy < w and maze[wx][wy] == 1:
            maze[wx][wy] = 0
            walls.extend([(wx + dx, wy + dy, wx, wy)
                          for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                          if 0 <= wx + dx < h and 0 <= wy + dy < w])
    return maze


# Génération d'une image PNG
def fancy_maze(bin_lab_2d, name):
    h, w = len(bin_lab_2d), len(bin_lab_2d[0])
    img = Image.new("1", (w, h))
    pixels = img.load()
    for i in range(h):
        for j in range(w):
            pixels[j, i] = 1 - bin_lab_2d[i][j]
    img.save(f"{name}.png")


# Lecture d'une image PNG
def unmaze(file_name):
    img = Image.open(file_name).convert("1")
    binary_map = []
    for y in range(img.height):
        for x in range(img.width):
            binary_map.append(1 - img.getpixel((x, y)))
    return {"maze": binary_map}


# Trouver des points aléatoires in et out
def get_random_in_out(lab_bin_2d):
    h, w = len(lab_bin_2d), len(lab_bin_2d[0])
    free_cells = [(i, j) for i in range(h) for j in range(w) if lab_bin_2d[i][j] == 0]
    return random.sample(free_cells, 2)


# Algorithme A*
def a_star(lab, in_out):
    start, goal = in_out
    h, w = len(lab), len(lab[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(goal[0] - start[0]) + abs(goal[1] - start[1])}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < h and 0 <= neighbor[1] < w and lab[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + abs(goal[0] - neighbor[0]) + abs(goal[1] - neighbor[1])
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []


# Validation du chemin
def check_path(path):
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        if abs(x1 - x2) + abs(y1 - y2) != 1:
            return False
    return True


# Exemple d'utilisation
if __name__ == "__main__":
    w, h = 10, 10
    maze = generate_prim(w, h)
    print_maze(maze)
    in_out = get_random_in_out(maze)
    print("\nWith In and Out:")
    print_maze(maze, in_out)
    path = a_star(maze, in_out)
    print("\nPath:")
    print_maze(maze, in_out, path)
    print("Path valid:", check_path(path))
    fancy_maze(maze, "maze")