import pygame
import json
import random
import sys
from collections import deque
import heapq # Added for A* priority queue

# Initialize Pygame
pygame.init()

# Constants
GRID_ROWS = 5
GRID_COLS = 10
CELL_SIZE = 80
WINDOW_WIDTH = GRID_COLS * CELL_SIZE
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE + 100
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
BROWN = (139, 69, 19)

# Global renderer placeholder (needed for execute_path visualization)
renderer = None

# Load team configuration
def load_config():
    try:
        with open('team_config.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("Error: team_config.json not found!")
        sys.exit(1)

# World class
class BangaloreWumpusWorld:
    def __init__(self, config):
        self.config = config
        self.seed = config['seed']
        random.seed(self.seed)

        # Initialize grid
        self.grid = [[{'type': 'empty', 'percepts': [], 'weight': random.randint(1,15)}
              for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

        # Agent starts at bottom-left diagonal
        self.agent_start = (0, GRID_ROWS - 1)
        self.agent_pos = list(self.agent_start)
        self.agent_path = []

        # Game state
        self.game_over = False
        self.game_won = False
        self.message = ""

        # Elements to avoid / pathfinding knowledge
        self.forbidden_cells = set() # To store cells that caused a reset (cows)

        # Generate world
        self._generate_world()

    def _generate_world(self):
        """Generate random world elements based on config"""
        num_traffic_lights = self.config['grid_config']['traffic_lights']
        num_cows = self.config['grid_config']['cows']
        num_pits = self.config['grid_config']['pits']

        # Generate available positions (exclude agent start)
        available_positions = [(x, y) for x in range(GRID_COLS) for y in range(GRID_ROWS)
                               if (x, y) != tuple(self.agent_start)]

        random.shuffle(available_positions)

        # Place traffic lights
        for i in range(num_traffic_lights):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'traffic_light'

        # Place cows
        for i in range(num_cows):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'cow'

        # Place pits
        for i in range(num_pits):
            if available_positions:
                pos = available_positions.pop()
                self.grid[pos[1]][pos[0]]['type'] = 'pit'

        # Place goal
        if available_positions:
            goal_pos = available_positions.pop()
            self.grid[goal_pos[1]][goal_pos[0]]['type'] = 'goal'
            self.goal_pos = goal_pos
        else:
            self.goal_pos = None
            self.message = "No space for Goal"

        # Generate percepts
        self._generate_percepts()

    def _generate_percepts(self):
        """Generate percepts for all cells based on adjacent elements"""
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                neighbors = self._get_neighbors(x, y)

                for nx, ny in neighbors:
                    cell_type = self.grid[ny][nx]['type']

                    if cell_type == 'pit':
                        if 'breeze' not in self.grid[y][x]['percepts']:
                            self.grid[y][x]['percepts'].append('breeze')

                    elif cell_type == 'cow':
                        if 'moo' not in self.grid[y][x]['percepts']:
                            self.grid[y][x]['percepts'].append('moo')

                    elif cell_type == 'traffic_light':
                        if 'light' not in self.grid[y][x]['percepts']:
                            self.grid[y][x]['percepts'].append('light')

    def _get_neighbors(self, x, y):
        """Get valid adjacent neighbors (no diagonals - up, down, left, right only)"""
        neighbors = []
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS:
                neighbors.append((nx, ny))

        return neighbors

    def move_agent(self, new_x, new_y):
        """Move agent to new position and handle interactions"""
        if self.game_over or self.game_won:
            return

        # Check bounds
        if not (0 <= new_x < GRID_COLS and 0 <= new_y < GRID_ROWS):
            return

        # Only allow orthogonal movement (no diagonals)
        dx = abs(new_x - self.agent_pos[0])
        dy = abs(new_y - self.agent_pos[1])
        if dx + dy != 1:
            if (new_x, new_y) != self.agent_start or tuple(self.agent_pos) != self.agent_start:
                return

        self.agent_pos = [new_x, new_y]
        
        # Add to path only if it's a new unique step
        if not self.agent_path or tuple(self.agent_pos) != self.agent_path[-1]:
            self.agent_path.append(tuple(self.agent_pos))

        cell_type = self.grid[new_y][new_x]['type']

        # Handle traffic light - simulate delay
        if cell_type == 'traffic_light':
            self.message = "Waiting at traffic signal..."
            self._simulate_traffic_delay()

        # Handle cow - reset to start
        elif cell_type == 'cow':
            self.message = "Moo! Cow encountered - returning to start! Recomputing path..."
            # Mark this cow cell as forbidden for future pathfinding
            self.forbidden_cells.add((new_x, new_y))
            self.agent_pos = list(self.agent_start)
            self.agent_path = []

        # Handle pit - game over (main loop handles auto-restart)
        elif cell_type == 'pit':
            self.message = "Game Over - Fell into a pit! Restarting..."
            self.game_over = True # Signal the main loop to restart

        # Handle goal - win
        elif cell_type == 'goal':
            self.message = "Goal Reached! You won!"
            self.game_won = True
        
        # Update message for empty cells
        elif cell_type == 'empty':
             self.message = "Clear path."


    def _simulate_traffic_delay(self):
        """Simulate traffic light delay using nested loop"""
        delay = 0
        for i in range(1000):
            for j in range(10000):
                delay += 1

    def get_current_percepts(self):
        """Get percepts at current agent position"""
        x, y = self.agent_pos
        return self.grid[y][x]['percepts']

    def find_path_astar(self):
        """
        Implementation of A* pathfinding algorithm
        """
        start = tuple(self.agent_pos)
        if not hasattr(self, 'goal_pos') or self.goal_pos is None:
            self.message = "Path Not Found - Goal position unknown."
            return None
        goal = self.goal_pos

        if start == goal:
            return [start]

        # Helper function for Manhattan Distance (Heuristic h(n))
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Helper function to get the cost of moving to cell (x, y)
        def get_cell_cost(x, y):
            cell = self.grid[y][x]
            cell_type = cell['type']
            base_cost = cell['weight']

            if (x, y) in self.forbidden_cells:
                return float('inf')

            if cell_type == 'pit':
                return float('inf')
            elif cell_type == 'cow':
                # Treat unvisited cow cells as infinite cost to avoid the reset trap
                return float('inf')
            elif cell_type == 'traffic_light':
                # Traffic lights add a wait penalty 
                return base_cost + 5
            else:
                # Empty/Goal cells use their base weight + 1 (minimum movement cost)
                return base_cost + 1

        # Priority Queue: Stores (f_score, g_score, x, y)
        open_set = []
        g_score = { (x, y): float('inf') for x in range(GRID_COLS) for y in range(GRID_ROWS) }
        g_score[start] = 0
        f_score = { (x, y): float('inf') for x in range(GRID_COLS) for y in range(GRID_ROWS) }
        f_score[start] = heuristic(start, goal)
        
        came_from = {} 

        # Push (f_score, g_score, node) to open_set
        heapq.heappush(open_set, (f_score[start], g_score[start], start))

        while open_set:
            current_f, current_g, current_node = heapq.heappop(open_set)
            cx, cy = current_node

            if current_node == goal:
                # Reconstruct path
                path = []
                while current_node in came_from:
                    path.append(current_node)
                    current_node = came_from[current_node]
                path.append(start)
                
                path.reverse()
                self.message = f"Optimal Path Found (Cost: {current_g})"
                return path

            for neighbor in self._get_neighbors(cx, cy):
                nx, ny = neighbor
                
                move_cost = get_cell_cost(nx, ny)

                if move_cost == float('inf'):
                    continue

                tentative_g_score = g_score[current_node] + move_cost

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    
                    heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))


        self.message = "Path Not Found - No safe or reachable path to goal."
        return None

    def execute_path(self, path):
        """Execute a computed path step by step"""
        if path is None:
            return

        current_pos_tuple = tuple(self.agent_pos)
        
        try:
            # Start from the next step if agent is found in the path
            start_index = path.index(current_pos_tuple)
            steps = path[start_index + 1:] 
        except ValueError:
            # Agent is not on the path (e.g., just reset). Start from the beginning.
            steps = path
            
        self.agent_path = [current_pos_tuple]


        for x, y in steps:
            is_cow_cell = self.grid[y][x]['type'] == 'cow'
            
            self.move_agent(x, y)
            
            if self.game_over or self.game_won:
                break
            
            # If a cow was hit, the agent resets and a replan is needed
            if is_cow_cell and tuple(self.agent_pos) == self.agent_start:
                print("Replanning A* path...")
                self.forbidden_cells.add((x, y)) 
                
                new_path = self.find_path_astar()
                if new_path:
                    # Recursively execute the new path
                    self.execute_path(new_path)
                return # Exit current execution/recursing execution
            
            # Pause for visualization
            pygame.time.wait(200) 
            
            # Rerender the screen to show movement
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
            
            global renderer
            renderer.render()
            
# Pygame rendering
class GameRenderer:
    def __init__(self, world):
        self.world = world
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Bangalore Wumpus World - AI CODEFIX 2025")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def draw_grid(self):
        """Draw the grid lines"""
        for x in range(0, WINDOW_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (x, 0), (x, WINDOW_HEIGHT - 100), 2)
        for y in range(0, WINDOW_HEIGHT - 100, CELL_SIZE):
            pygame.draw.line(self.screen, BLACK, (0, y), (WINDOW_WIDTH, y), 2)

    def draw_cell_contents(self):
        """Draw contents of each cell"""
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                cell = self.world.grid[y][x]
                px = x * CELL_SIZE
                py = y * CELL_SIZE
                
                # Highlight forbidden cells (for replanning visualization)
                if (x, y) in self.world.forbidden_cells:
                    pygame.draw.rect(self.screen, (100, 0, 0), (px, py, CELL_SIZE, CELL_SIZE))

                # Draw cell type
                if cell['type'] == 'traffic_light':
                    pygame.draw.circle(self.screen, RED, (px + CELL_SIZE//2, py + CELL_SIZE//2), 20)
                    text = self.small_font.render("SIGNAL", True, WHITE)
                    self.screen.blit(text, (px + 15, py + 55))

                elif cell['type'] == 'cow':
                    pygame.draw.rect(self.screen, BROWN, (px + 20, py + 20, 40, 40))
                    text = self.small_font.render("COW", True, WHITE)
                    self.screen.blit(text, (px + 25, py + 30))

                elif cell['type'] == 'pit':
                    pygame.draw.circle(self.screen, BLACK, (px + CELL_SIZE//2, py + CELL_SIZE//2), 25)
                    text = self.small_font.render("PIT", True, WHITE)
                    self.screen.blit(text, (px + 28, py + 30))

                elif cell['type'] == 'goal':
                    pygame.draw.rect(self.screen, GREEN, (px + 15, py + 15, 50, 50))
                    text = self.small_font.render("GOAL", True, BLACK)
                    self.screen.blit(text, (px + 20, py + 30))

                # Draw percepts (small indicators)
                percept_y_offset = 10
                if 'breeze' in cell['percepts']:
                    text = self.small_font.render("~", True, BLUE)
                    self.screen.blit(text, (px + 5, py + percept_y_offset))
                    percept_y_offset += 15

                if 'moo' in cell['percepts']:
                    text = self.small_font.render("M", True, BROWN)
                    self.screen.blit(text, (px + 5, py + percept_y_offset))
                    percept_y_offset += 15

                if 'light' in cell['percepts']:
                    text = self.small_font.render("L", True, ORANGE)
                    self.screen.blit(text, (px + 5, py + percept_y_offset))

    def draw_agent(self):
        """Draw the agent"""
        x, y = self.world.agent_pos
        px = x * CELL_SIZE + CELL_SIZE // 2
        py = y * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(self.screen, YELLOW, (px, py), 15)
        pygame.draw.circle(self.screen, BLACK, (px, py), 15, 2)

        # Draw eyes
        pygame.draw.circle(self.screen, BLACK, (px - 5, py - 3), 3)
        pygame.draw.circle(self.screen, BLACK, (px + 5, py - 3), 3)

    def draw_info(self):
        """Draw info panel at bottom"""
        info_y = WINDOW_HEIGHT - 100
        pygame.draw.rect(self.screen, GRAY, (0, info_y, WINDOW_WIDTH, 100))

        # Current position
        pos_text = self.font.render(f"Position: {self.world.agent_pos}", True, BLACK)
        self.screen.blit(pos_text, (10, info_y + 10))

        # Percepts
        percepts = self.world.get_current_percepts()
        percept_text = self.font.render(f"Percepts: {', '.join(percepts) if percepts else 'None'}", True, BLACK)
        self.screen.blit(percept_text, (10, info_y + 35))

        # Message
        msg_text = self.font.render(self.world.message, True, RED if self.world.game_over else GREEN)
        self.screen.blit(msg_text, (10, info_y + 60))

    def render(self):
        """Main render function"""
        self.screen.fill(WHITE)
        self.draw_grid()
        self.draw_cell_contents()
        self.draw_agent()
        self.draw_info()
        pygame.display.flip()
        self.clock.tick(FPS)

# Main game loop
def main():
    global renderer # Use the global renderer placeholder
    
    config = load_config()
    world = BangaloreWumpusWorld(config)
    renderer = GameRenderer(world)

    print("=== Bangalore Wumpus World ===")
    print(f"Team ID: {config['team_id']}")
    print(f"Agent Start: {world.agent_start}")
    print(f"Goal Position: {world.goal_pos}")
    print("\nControls:")
    print("- Arrow keys: Manual movement")
    print("- SPACE: Execute A* pathfinding")
    print("- R: Reset world")
    print("- ESC: Quit")

    running = True
    while running:
        
        # --- LOGIC: CHECK FOR GAME OVER AND AUTO-RESTART ---
        if world.game_over and not world.game_won:
            # Game is over (pit collision). Wait briefly then re-initialize.
            pygame.time.wait(1000) 
            world = BangaloreWumpusWorld(config)
            renderer.world = world # Update the renderer's reference
            world.message = "Game restarted automatically."
        # --------------------------------------------------

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                current_x, current_y = world.agent_pos
                new_x, new_y = current_x, current_y
                moved = False
                
                # Prevent any new moves if the game is already finished/waiting for restart
                if world.game_over or world.game_won:
                    continue

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    # Manual Reset world
                    world = BangaloreWumpusWorld(config)
                    renderer.world = world

                elif event.key == pygame.K_SPACE:
                    # Call A* pathfinding
                    print("\n=== Executing A* Pathfinding ===")
                    path = world.find_path_astar()
                    if path:
                        print(f"Path found: {path}")
                        world.execute_path(path)
                    else:
                        print(world.message)
                
                # --- MANUAL MOVEMENT LOGIC ---
                elif event.key == pygame.K_LEFT:
                    new_x -= 1
                    moved = True
                elif event.key == pygame.K_RIGHT:
                    new_x += 1
                    moved = True
                elif event.key == pygame.K_UP:
                    new_y -= 1 
                    moved = True
                elif event.key == pygame.K_DOWN:
                    new_y += 1 
                    moved = True
                
                if moved:
                    world.move_agent(new_x, new_y)
                # --- END MANUAL MOVEMENT LOGIC ---
                

        renderer.render()

    pygame.quit()

if __name__ == "__main__":
    main()