import os

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from pyvirtualdisplay import Display
import pygame
import numpy as np

# Initialize Pygame
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['DISPLAY'] = 'xvfb'
disp = Display(visible=0, backend='xvfb')
disp.start()
pygame.init()

# Tensor dimensions (N x M)
N, M = 10, 10  # Example dimensions, modify as needed

# Window size
width, height = 600, 600
cell_width, cell_height = width // M, height // N

# Create window
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Tensor Map Renderer')

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
font = pygame.font.Font(None, 12)

# Example tensors
obstacle_tensor = np.random.randint(0, 2, (N, M))
agent_tensor = np.random.randint(0, 5, (N, M))
goal_tensor = np.zeros((N, M))

# remove agents that are currently on obstacles
for i in range(N):
    for j in range(M):
        if agent_tensor[i, j] > 0 and obstacle_tensor[i, j] > 0:
            agent_tensor[i, j] = 0

# Assign unique indices to agents
agent_indices = np.zeros_like(agent_tensor)
current_index = 1
for i in range(N):
    for j in range(M):
        if agent_tensor[i, j] != 0:
            agent_indices[i, j] = current_index
            current_index += 1
            # now generate a goal for each agent
            invalid_goal = True
            counter = 0
            while invalid_goal and counter < 20:
                rand_row = np.random.randint(0, N)
                rand_col = np.random.randint(0, M)
                if agent_tensor[rand_row, rand_col] == 0 and obstacle_tensor[rand_row, rand_col] == 0:
                    goal_tensor[rand_row, rand_col] = agent_indices[i, j]
                    invalid_goal = False
                counter += 1
            
def draw_triangle(center, size, direction, color):
    x, y = center
    if direction == 1:  # Up
        points = [(x, y - size), (x - size, y + size), (x + size, y + size)]
    elif direction == 2:  # Right
        points = [(x + size, y), (x - size, y - size), (x - size, y + size)]
    elif direction == 3:  # Down
        points = [(x, y + size), (x - size, y - size), (x + size, y - size)]
    elif direction == 4:  # Left
        points = [(x - size, y), (x + size, y - size), (x + size, y + size)]
    pygame.draw.polygon(screen, color, points)

def draw_map():
    for i in range(N):
        for j in range(M):
            rect = pygame.Rect(j * cell_width, i * cell_height, cell_width, cell_height)
            pygame.draw.rect(screen, WHITE, rect, 1)

            # Draw obstacles
            if obstacle_tensor[i, j] == 1:
                pygame.draw.rect(screen, BLACK, rect)

                      # Draw agents and their indices
            if agent_tensor[i, j] != 0:
                color = BLUE
                draw_triangle(rect.center, min(cell_width, cell_height) // 4, agent_tensor[i, j], color)
                index_text = font.render(str(agent_indices[i, j]), True, WHITE)
                text_rect = index_text.get_rect(center=rect.center)
                screen.blit(index_text, text_rect)

            # Draw goals
            if goal_tensor[i, j] != 0:
                goal_text = font.render(str(goal_tensor[i, j]), True, BLACK)
                goal_rect = goal_text.get_rect(center=rect.center)
                pygame.draw.circle(screen, WHITE, rect.center, min(cell_width, cell_height) // 4)
                screen.blit(goal_text, goal_rect)

# Video writing setup using OpenCV
video_filename = 'output.mp4'
movie_clips = []
# Generate frames and write to video

for _ in range(100):  # Number of frames
    screen.fill(WHITE)
    draw_map()
    frame = pygame.surfarray.array3d(screen)
    movie_clips.append(np.transpose(frame, (1, 0, 2)))

clip = ImageSequenceClip(movie_clips, fps=15)
clip.write_videofile(os.path.join(os.getcwd(), video_filename), threads=4)
pygame.quit()