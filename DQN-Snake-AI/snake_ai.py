"""
Snake AI Environment with Enhanced State Representation and Advanced Reward Engineering (Phase 3).
Features:
- Configurable State Modes: "standard" (11 features) or "enhanced" (24 features)
- Configurable Reward Modes: "phase1" (baseline) or "advanced" (PBRS + trap avoidance)
- Lookahead raycasts and wall proximity
- Flood-fill accessible space and trap risk estimation
- Visual overlay support for A* paths and Q-values
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pygame
import torch

from config import (
    BLOCK_SIZE,
    COLOR_BLACK,
    COLOR_BLUE,
    COLOR_CYAN,
    COLOR_DARK_GRAY,
    COLOR_DARK_GREEN,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_WHITE,
    COLOR_YELLOW,
    GRID_HEIGHT,
    GRID_WIDTH,
    RENDER_FPS,
    REWARD_CLOSER,
    REWARD_COLLISION,
    REWARD_FARTHER,
    REWARD_FOOD,
    REWARD_MODE,
    REWARD_SPACE_FACTOR,
    REWARD_STEP,
    REWARD_TRAP_PENALTY,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    STARVATION_BASE_STEPS,
    STARVATION_FACTOR,
    STATE_MODE,
    STATE_SIZE_ENHANCED,
    STATE_SIZE_STANDARD,
    TOTAL_GRID_CELLS,
)


class SnakeGameAI:
    """
    Snake environment supporting standard (11-D) and enhanced (24-D) state representations,
    baseline and advanced reward modes, and decision overlay rendering.
    """

    def __init__(
        self,
        display: bool = True,
        width: int = SCREEN_WIDTH,
        height: int = SCREEN_HEIGHT,
        block_size: int = BLOCK_SIZE,
        fps: int = RENDER_FPS,
        state_mode: str = STATE_MODE,
        reward_mode: str = REWARD_MODE,
    ):
        self.display = display
        self.width = width
        self.height = height
        self.w = width
        self.h = height
        self.block_size = block_size
        self.fps = fps
        self.state_mode = state_mode.lower()
        self.reward_mode = reward_mode.lower()

        self.max_diagonal = math.hypot(self.width, self.height)
        self.max_manhattan = self.width + self.height
        self.max_blocks = max(self.width // self.block_size, self.height // self.block_size)
        self.grid_width = self.width // self.block_size
        self.grid_height = self.height // self.block_size
        self.total_grid_cells = self.grid_width * self.grid_height

        self.screen = None
        self.clock = None

        if self.display:
            if not pygame.get_init():
                pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake AI - Phase 3 Intelligence")
            self.clock = pygame.time.Clock()

        self.reset()

    @property
    def state_size(self) -> int:
        """Return dimensionality of the observation space."""
        return STATE_SIZE_ENHANCED if self.state_mode == "enhanced" else STATE_SIZE_STANDARD

    def reset(self) -> np.ndarray:
        """Reset game state to initial configuration."""
        self.snake = [
            (100, 100),
            (80, 100),
            (60, 100),
        ]
        self.direction = "RIGHT"
        self.score = 0
        self.food_eaten = 0
        self.game_over = False
        self.frame_iteration = 0
        self.steps_without_food = 0
        self.total_reward = 0.0
        self.reward_components: Dict[str, float] = {
            "food": 0.0,
            "shaping": 0.0,
            "trap": 0.0,
            "collision": 0.0,
            "step": 0.0,
        }
        self.food = self.generate_food()
        return self.get_state()

    def generate_food(self) -> Tuple[int, int]:
        """Generate food coordinates not occupied by snake body."""
        total_slots = (self.width // self.block_size) * (self.height // self.block_size)
        if len(self.snake) >= total_slots:
            return (-1, -1)

        while True:
            food = (
                random.randrange(0, self.width, self.block_size),
                random.randrange(0, self.height, self.block_size),
            )
            if food not in self.snake:
                return food

    def _get_distance_to_food(self, point: Tuple[int, int]) -> float:
        """Calculate Euclidean distance to food."""
        return math.hypot(point[0] - self.food[0], point[1] - self.food[1])

    def _get_candidate_point(
        self,
        direction: Union[str, Tuple[int, int]],
        head: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """
        Compute candidate (x, y) coordinates for one step given direction and head position.
        Supports:
          - direction as offset tuple (dx, dy) or cardinal string ("UP", "DOWN", "LEFT", "RIGHT")
          - head as coordinate tuple (hx, hy) or None (defaults to self.snake[0])
        """
        # Resolve head coordinate
        if head is None:
            hx, hy = self.snake[0] if self.snake else (0, 0)
        elif isinstance(head, (tuple, list)) and len(head) == 2:
            hx, hy = head[0], head[1]
        else:
            # Handle potential swapped arguments (head, direction)
            if isinstance(direction, (tuple, list)) and len(direction) == 2:
                hx, hy = direction[0], direction[1]
                direction = head
            else:
                hx, hy = self.snake[0] if self.snake else (0, 0)

        # Resolve direction offset (dx, dy)
        if isinstance(direction, str):
            d = direction.upper()
            if d == "UP":
                dx, dy = 0, -self.block_size
            elif d == "DOWN":
                dx, dy = 0, self.block_size
            elif d == "LEFT":
                dx, dy = -self.block_size, 0
            elif d == "RIGHT":
                dx, dy = self.block_size, 0
            else:
                dx, dy = 0, 0
        elif isinstance(direction, (tuple, list)) and len(direction) == 2:
            dx, dy = direction[0], direction[1]
        else:
            dx, dy = 0, 0

        return (hx + dx, hy + dy)

    get_candidate_point = _get_candidate_point

    def check_collision(
        self, point_or_direction: Optional[Union[str, Tuple[int, int]]] = None
    ) -> bool:
        """Check collision with walls or body segments (excluding vacating tail)."""
        if point_or_direction is None:
            point = self.snake[0]
            x, y = point
        elif isinstance(point_or_direction, str):
            x, y = self.snake[0]
            if point_or_direction == "UP":
                y -= self.block_size
            elif point_or_direction == "DOWN":
                y += self.block_size
            elif point_or_direction == "LEFT":
                x -= self.block_size
            elif point_or_direction == "RIGHT":
                x += self.block_size
            point = (x, y)
        else:
            point = point_or_direction
            x, y = point

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        if len(self.snake) <= 1:
            return False

        if point == self.snake[0]:
            # Checking head collision with own body segments (excluding vacating tail)
            body_to_check = self.snake[1:-1] if len(self.snake) > 2 else []
        else:
            # Checking candidate point with current body (excluding vacating tail)
            body_to_check = self.snake[:-1]

        return point in body_to_check

    def _raycast(self, direction: str) -> int:
        """
        Count the number of contiguous free grid blocks in a cardinal direction
        before striking a wall or snake body segment.
        """
        x, y = self.snake[0]
        dx, dy = 0, 0
        if direction == "UP":
            dy = -self.block_size
        elif direction == "DOWN":
            dy = self.block_size
        elif direction == "LEFT":
            dx = -self.block_size
        elif direction == "RIGHT":
            dx = self.block_size

        count = 0
        body_set = set(self.snake[:-1])
        while True:
            x += dx
            y += dy
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                break
            if (x, y) in body_set:
                break
            count += 1
        return count

    def _flood_fill_space(
        self, start_point: Optional[Tuple[int, int]] = None, max_depth: int = 150
    ) -> int:
        """BFS counting accessible connected empty blocks from a given screen point."""
        from collections import deque

        if start_point is None:
            start_point = self.snake[0] if self.snake else (0, 0)

        x, y = start_point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return 0

        # Define obstacle set
        if len(self.snake) > 2:
            obstacles = set(self.snake[1:-1]) if start_point == self.snake[0] else set(self.snake[:-1])
        else:
            obstacles = set()

        if start_point in obstacles:
            return 0

        visited = {start_point}
        queue = deque([start_point])
        count = 0

        while queue and count < max_depth:
            curr_x, curr_y = queue.popleft()
            count += 1

            for dx, dy in [(0, -self.block_size), (self.block_size, 0), (0, self.block_size), (-self.block_size, 0)]:
                nx, ny = curr_x + dx, curr_y + dy
                neighbor = (nx, ny)
                if (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and neighbor not in visited
                    and neighbor not in obstacles
                ):
                    visited.add(neighbor)
                    queue.append(neighbor)

        return count

    def move(self, action: Union[int, list, np.ndarray, torch.Tensor]) -> Tuple[float, bool, int]:
        """
        Execute one action step in the environment.
        Supports both baseline (Phase 1) and advanced (Phase 3) reward engineering.
        """
        self.frame_iteration += 1
        self.steps_without_food += 1

        if isinstance(action, (list, tuple, np.ndarray, torch.Tensor)):
            if hasattr(action, "tolist"):
                action = action.tolist()
            if isinstance(action, list) and len(action) == 3:
                action = action.index(max(action))
            else:
                action = int(action)

        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        idx = directions.index(self.direction)

        if action == 1:
            idx = (idx + 1) % 4
        elif action == 2:
            idx = (idx - 1) % 4

        self.direction = directions[idx]

        old_head = self.snake[0]
        x, y = old_head

        if self.direction == "UP":
            y -= self.block_size
        elif self.direction == "DOWN":
            y += self.block_size
        elif self.direction == "LEFT":
            x -= self.block_size
        elif self.direction == "RIGHT":
            x += self.block_size

        new_head = (x, y)

        # 1. Collision Check
        if self.check_collision(new_head):
            self.game_over = True
            reward = REWARD_COLLISION
            self.total_reward += reward
            self.reward_components["collision"] += reward
            return reward, True, self.score

        old_dist = self._get_distance_to_food(old_head)
        self.snake.insert(0, new_head)

        # 2. Starvation / Infinite Loop Limit
        max_allowed_steps = max(STARVATION_BASE_STEPS, STARVATION_FACTOR * len(self.snake))
        if self.steps_without_food > max_allowed_steps:
            self.game_over = True
            reward = REWARD_COLLISION
            self.total_reward += reward
            self.reward_components["collision"] += reward
            return reward, True, self.score

        # 3. Reward Calculation
        if new_head == self.food:
            self.score += 1
            self.food_eaten += 1
            self.steps_without_food = 0
            self.food = self.generate_food()
            reward = REWARD_FOOD
            self.reward_components["food"] += reward
        else:
            self.snake.pop()
            new_dist = self._get_distance_to_food(new_head)

            if self.reward_mode == "advanced":
                # Phase 3 & 4: Potential-Based Reward Shaping (PBRS)
                # Strong gradient: +0.1 for moving 1 block closer, -0.1 for moving 1 block farther
                dist_delta = (old_dist - new_dist) / self.block_size
                shaping_reward = 0.1 * dist_delta
                reward = REWARD_STEP + shaping_reward
                self.reward_components["shaping"] += shaping_reward

                # Trap Penalty: Check if entered an enclosed pocket strictly smaller than body
                # Only apply when snake has accumulated non-trivial length (>= 4)
                free_space = self._flood_fill_space(new_head, max_depth=len(self.snake) + 10)
                if len(self.snake) >= 4 and free_space < len(self.snake):
                    trap_penalty = REWARD_TRAP_PENALTY
                    reward += trap_penalty
                    self.reward_components["trap"] += trap_penalty
            else:
                # Phase 1: Baseline reward shaping
                reward = REWARD_STEP
                if new_dist < old_dist:
                    reward += REWARD_CLOSER
                elif new_dist > old_dist:
                    reward += REWARD_FARTHER
                self.reward_components["step"] += reward

        self.total_reward += reward
        return reward, False, self.score

    def get_state(self) -> np.ndarray:
        """
        Extract observation state vector based on configured state_mode.
        - "standard": 11 binary/integer features (Phase 1 Baseline).
        - "enhanced": 24 normalized features (Phase 3 Intelligence).
        """
        if self.state_mode == "standard":
            return self._get_standard_state()
        else:
            return self._get_enhanced_state()

    def _get_standard_state(self) -> np.ndarray:
        """Extract original 11-feature baseline state vector."""
        head_x, head_y = self.snake[0]
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        idx = directions.index(self.direction)

        danger_straight = self.check_collision(self.direction)
        danger_right = self.check_collision(directions[(idx + 1) % 4])
        danger_left = self.check_collision(directions[(idx - 1) % 4])

        food_left = self.food[0] < head_x
        food_right = self.food[0] > head_x
        food_up = self.food[1] < head_y
        food_down = self.food[1] > head_y

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),
            int(self.direction == "UP"),
            int(self.direction == "RIGHT"),
            int(self.direction == "DOWN"),
            int(self.direction == "LEFT"),
            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down),
        ]
        return np.array(state, dtype=np.float32)

    def _get_enhanced_state(self) -> np.ndarray:
        """
        Extract 24-dimensional normalized enhanced state vector (Phase 3).
        
        Features:
            [0..2]   Immediate Danger (Straight, Right, Left) -> 0.0 or 1.0
            [3..6]   Current Direction (UP, RIGHT, DOWN, LEFT) -> 0.0 or 1.0
            [7..10]  Food Relative Direction (Left, Right, Up, Down) -> 0.0 or 1.0
            [11..12] Normalized Food Distance (Euclidean & Manhattan) -> [0.0, 1.0]
            [13..16] Wall Proximity (Distance to Up, Right, Down, Left wall) -> [0.0, 1.0]
            [17..19] Lookahead Free Blocks Raycast (Straight, Right, Left) -> [0.0, 1.0]
            [20]     Snake Length Ratio -> [0.0, 1.0]
            [21..22] Tail Relative Offset (dx, dy normalized) -> [0.0, 1.0]
            [23]     Surrounding Accessible Space Ratio (BFS flood fill) -> [0.0, 1.0]
        """
        head_x, head_y = self.snake[0]
        tail_x, tail_y = self.snake[-1]
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        idx = directions.index(self.direction)

        # 1. Immediate Danger
        dir_straight = self.direction
        dir_right = directions[(idx + 1) % 4]
        dir_left = directions[(idx - 1) % 4]

        danger_straight = float(self.check_collision(dir_straight))
        danger_right = float(self.check_collision(dir_right))
        danger_left = float(self.check_collision(dir_left))

        # 2. Current Heading
        d_up = float(self.direction == "UP")
        d_right = float(self.direction == "RIGHT")
        d_down = float(self.direction == "DOWN")
        d_left = float(self.direction == "LEFT")

        # 3. Relative Food Quadrant
        f_left = float(self.food[0] < head_x)
        f_right = float(self.food[0] > head_x)
        f_up = float(self.food[1] < head_y)
        f_down = float(self.food[1] > head_y)

        # 4. Normalized Food Distance
        eucl_dist = math.hypot(head_x - self.food[0], head_y - self.food[1]) / self.max_diagonal
        manh_dist = (abs(head_x - self.food[0]) + abs(head_y - self.food[1])) / self.max_manhattan

        # 5. Wall Proximity
        wall_up = head_y / max(1, self.height - self.block_size)
        wall_right = (self.width - self.block_size - head_x) / max(1, self.width - self.block_size)
        wall_down = (self.height - self.block_size - head_y) / max(1, self.height - self.block_size)
        wall_left = head_x / max(1, self.width - self.block_size)

        # 6. Raycast Obstacle Proximity
        ray_straight = min(1.0, self._raycast(dir_straight) / self.max_blocks)
        ray_right = min(1.0, self._raycast(dir_right) / self.max_blocks)
        ray_left = min(1.0, self._raycast(dir_left) / self.max_blocks)

        # 7. Snake Length Ratio
        snake_len_norm = min(1.0, len(self.snake) / TOTAL_GRID_CELLS)

        # 8. Normalized Tail Offset
        tail_dx = (tail_x - head_x) / (2.0 * self.width) + 0.5
        tail_dy = (tail_y - head_y) / (2.0 * self.height) + 0.5

        # 9. Accessible Free Space Ratio
        free_space = self._flood_fill_space(self.snake[0], max_depth=100)
        free_space_norm = min(1.0, free_space / 100.0)

        state = [
            danger_straight,
            danger_right,
            danger_left,
            d_up,
            d_right,
            d_down,
            d_left,
            f_left,
            f_right,
            f_up,
            f_down,
            eucl_dist,
            manh_dist,
            wall_up,
            wall_right,
            wall_down,
            wall_left,
            ray_straight,
            ray_right,
            ray_left,
            snake_len_norm,
            tail_dx,
            tail_dy,
            free_space_norm,
        ]
        return np.array(state, dtype=np.float32)

    def draw(
        self,
        record: int = 0,
        path: Optional[List[Tuple[int, int]]] = None,
        q_values: Optional[List[float]] = None,
        selected_action: Optional[int] = None,
        astar_action: Optional[int] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """
        Render game frame with decision telemetry overlay (HUD, A* path, Q-values).
        """
        if not self.display or self.screen is None:
            return

        pygame.event.pump()
        self.screen.fill(COLOR_BLACK)

        # Draw A* Path Overlay if provided
        if path and len(path) > 1:
            for pt in path:
                # pt is in grid coordinates; convert to screen
                gx, gy = pt
                px = gx * self.block_size + self.block_size // 4
                py = gy * self.block_size + self.block_size // 4
                pw = self.block_size // 2
                pygame.draw.rect(self.screen, COLOR_CYAN, (px, py, pw, pw), border_radius=2)

        # Draw Snake
        for i, segment in enumerate(self.snake):
            color = COLOR_GREEN if i == 0 else COLOR_DARK_GREEN
            pygame.draw.rect(
                self.screen,
                color,
                (segment[0], segment[1], self.block_size, self.block_size),
                border_radius=3,
            )

        # Draw Food
        if self.food[0] >= 0 and self.food[1] >= 0:
            pygame.draw.rect(
                self.screen,
                COLOR_RED,
                (self.food[0], self.food[1], self.block_size, self.block_size),
                border_radius=4,
            )

        # Render HUD Overlay Text
        font = pygame.font.SysFont("arial", 18)
        score_surf = font.render(f"Score: {self.score} | Record: {record}", True, COLOR_WHITE)
        self.screen.blit(score_surf, (10, 8))

        # Render Q-Values and Decision HUD if available
        if q_values is not None and len(q_values) == 3:
            actions_str = ["Straight", "Right", "Left"]
            hud_y = 30
            for i, (act_name, q_val) in enumerate(zip(actions_str, q_values)):
                is_selected = (selected_action == i)
                color = COLOR_YELLOW if is_selected else COLOR_GRAY
                marker = "-> " if is_selected else "   "
                q_text = font.render(f"{marker}{act_name}: {q_val:+.2f}", True, color)
                self.screen.blit(q_text, (10, hud_y + i * 18))

            if astar_action is not None and astar_action < len(actions_str):
                astar_text = font.render(f"A* Suggests: {actions_str[astar_action]}", True, COLOR_CYAN)
                self.screen.blit(astar_text, (10, hud_y + 3 * 18))

            if epsilon is not None:
                eps_text = font.render(f"Epsilon: {epsilon:.3f}", True, COLOR_WHITE)
                self.screen.blit(eps_text, (self.width - 140, 8))

        pygame.display.flip()
        if self.clock is not None and self.fps > 0:
            self.clock.tick(self.fps)
