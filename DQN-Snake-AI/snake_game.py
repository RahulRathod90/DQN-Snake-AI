"""
Manual Human-Playable Snake Game using Pygame.
Controlled using Arrow keys (UP, DOWN, LEFT, RIGHT).
"""

import random
import pygame

from config import (
    BLOCK_SIZE,
    COLOR_BLACK,
    COLOR_DARK_GREEN,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_WHITE,
    RENDER_FPS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game - DQN AI Project (Manual Play)")
clock = pygame.time.Clock()

# Snake initial position
snake = [(100, 100), (80, 100), (60, 100)]
direction = "RIGHT"

# Food initial position
food = (
    random.randrange(0, SCREEN_WIDTH, BLOCK_SIZE),
    random.randrange(0, SCREEN_HEIGHT, BLOCK_SIZE),
)

score = 0
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != "DOWN":
                direction = "UP"
            elif event.key == pygame.K_DOWN and direction != "UP":
                direction = "DOWN"
            elif event.key == pygame.K_LEFT and direction != "RIGHT":
                direction = "LEFT"
            elif event.key == pygame.K_RIGHT and direction != "LEFT":
                direction = "RIGHT"

    # Compute new head position
    x, y = snake[0]
    if direction == "UP":
        y -= BLOCK_SIZE
    elif direction == "DOWN":
        y += BLOCK_SIZE
    elif direction == "LEFT":
        x -= BLOCK_SIZE
    elif direction == "RIGHT":
        x += BLOCK_SIZE

    new_head = (x, y)

    # Check wall or self collision
    if (
        x < 0
        or x >= SCREEN_WIDTH
        or y < 0
        or y >= SCREEN_HEIGHT
        or (len(snake) > 2 and new_head in snake[:-1])
    ):
        print(f"Game Over! Final Score: {score}")
        running = False
    else:
        snake.insert(0, new_head)

        # Eat food
        if new_head == food:
            score += 1
            while True:
                food = (
                    random.randrange(0, SCREEN_WIDTH, BLOCK_SIZE),
                    random.randrange(0, SCREEN_HEIGHT, BLOCK_SIZE),
                )
                if food not in snake:
                    break
        else:
            snake.pop()

    # Draw everything
    screen.fill(COLOR_BLACK)

    for i, segment in enumerate(snake):
        color = COLOR_GREEN if i == 0 else COLOR_DARK_GREEN
        pygame.draw.rect(screen, color, (segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE))

    pygame.draw.rect(screen, COLOR_RED, (food[0], food[1], BLOCK_SIZE, BLOCK_SIZE))

    font = pygame.font.SysFont("arial", 20)
    score_surf = font.render(f"Score: {score}", True, COLOR_WHITE)
    screen.blit(score_surf, (10, 10))

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
