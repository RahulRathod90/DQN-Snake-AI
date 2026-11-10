"""
A* Pathfinding, Safe Movement, and Trap Detection Engine for Snake AI (Phase 3).
Features:
- Grid-based A* search with Manhattan distance heuristic
- Dynamic obstacle detection (walls + snake body)
- Flood fill accessible space estimation (trap & bottleneck detection)
- Safe-path validation with tail reachability verification
- Tail-chasing / longest-path survival fallback
- Standalone Pure A* Agent (AStarAgent)
- DQN Action Safety Shield for hybrid DQN + A* execution
"""

import heapq
import math
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from config import BLOCK_SIZE, SCREEN_HEIGHT, SCREEN_WIDTH


def to_grid(point: Tuple[int, int], block_size: int = BLOCK_SIZE) -> Tuple[int, int]:
    """Convert screen coordinate (x, y) to grid coordinate (gx, gy)."""
    return (point[0] // block_size, point[1] // block_size)


def to_screen(grid_pos: Tuple[int, int], block_size: int = BLOCK_SIZE) -> Tuple[int, int]:
    """Convert grid coordinate (gx, gy) to screen coordinate (x, y)."""
    return (grid_pos[0] * block_size, grid_pos[1] * block_size)


class PathFinder:
    """
    Grid-based A* search and spatial safety analysis for Snake.
    Grid dimensions: W = SCREEN_WIDTH // BLOCK_SIZE, H = SCREEN_HEIGHT // BLOCK_SIZE.
    """

    def __init__(
        self,
        width: int = SCREEN_WIDTH,
        height: int = SCREEN_HEIGHT,
        block_size: int = BLOCK_SIZE,
    ):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.grid_w = width // block_size
        self.grid_h = height // block_size

    def is_valid_cell(self, cell: Tuple[int, int]) -> bool:
        """Check if grid cell is within screen boundaries."""
        gx, gy = cell
        return 0 <= gx < self.grid_w and 0 <= gy < self.grid_h

    def get_neighbors(
        self, cell: Tuple[int, int], obstacles: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Return valid adjacent cells (Up, Right, Down, Left) not in obstacles."""
        gx, gy = cell
        candidates = [(gx, gy - 1), (gx + 1, gy), (gx, gy + 1), (gx - 1, gy)]
        return [c for c in candidates if self.is_valid_cell(c) and c not in obstacles]

    def a_star(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacles: Set[Tuple[int, int]],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Classic A* search finding the shortest obstacle-free path from start to goal.
        Uses Manhattan distance heuristic.
        
        Returns:
            List of grid coordinates from start to goal, or None if unreachable.
        """
        if start == goal:
            return [start]

        if not self.is_valid_cell(start) or not self.is_valid_cell(goal):
            return None

        # Priority queue stores (f_score, g_score, current_cell)
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        f_score: Dict[Tuple[int, int], int] = {start: heuristic(start, goal)}
        closed_set: Set[Tuple[int, int]] = set()

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            closed_set.add(current)

            for neighbor in self.get_neighbors(current, obstacles):
                if neighbor in closed_set:
                    continue

                tentative_g = current_g + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, tentative_g, neighbor))

        return None

    def flood_fill_free_space(
        self, start: Tuple[int, int], obstacles: Set[Tuple[int, int]], max_depth: int = 400
    ) -> int:
        """
        Breadth-First Search (BFS) calculating the number of connected accessible empty cells.
        Used to detect enclosed traps and narrow corridors.
        """
        if not self.is_valid_cell(start) or start in obstacles:
            return 0

        visited: Set[Tuple[int, int]] = {start}
        queue: deque[Tuple[int, int]] = deque([start])
        count = 0

        while queue and count < max_depth:
            curr = queue.popleft()
            count += 1

            for neighbor in self.get_neighbors(curr, obstacles):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return count

    def is_path_safe(
        self,
        path: List[Tuple[int, int]],
        snake_body_grid: List[Tuple[int, int]],
    ) -> bool:
        """
        Verify if following the path leaves the snake in a safe position.
        Crucial Snake Rule: After reaching the food, the snake's tail must remain reachable.
        If the tail is unreachable, taking this food traps the snake in a dead end.
        """
        if not path or len(path) < 2:
            return False

        # Simulate snake body after traversing path
        simulated_body = list(snake_body_grid)
        for step in path[1:]:
            simulated_body.insert(0, step)
            # Food eaten at final step: do not pop on last step
            if step != path[-1]:
                simulated_body.pop()

        new_head = simulated_body[0]
        new_tail = simulated_body[-1]

        # Check if tail is reachable from new head
        obstacles = set(simulated_body[:-1])
        tail_path = self.a_star(new_head, new_tail, obstacles)
        return tail_path is not None

    def find_tail_chasing_path(
        self,
        head: Tuple[int, int],
        snake_body_grid: List[Tuple[int, int]],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Survival fallback: Find the longest safe path to the snake's tail.
        Because the tail moves forward every step, chasing the tail guarantees survival
        until an open safe route to food becomes available.
        """
        tail = snake_body_grid[-1]
        obstacles = set(snake_body_grid[:-1])
        return self.a_star(head, tail, obstacles)

    def grid_step_to_action(
        self,
        current_head_grid: Tuple[int, int],
        next_cell_grid: Tuple[int, int],
        current_direction: str,
    ) -> int:
        """
        Translate a move to an adjacent grid cell into a DQN relative action:
            0 = Straight, 1 = Turn Right, 2 = Turn Left.
        """
        hx, hy = current_head_grid
        nx, ny = next_cell_grid

        # Determine target cardinal direction
        if ny < hy:
            target_dir = "UP"
        elif ny > hy:
            target_dir = "DOWN"
        elif nx < hx:
            target_dir = "LEFT"
        elif nx > hx:
            target_dir = "RIGHT"
        else:
            return 0  # Same cell

        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        cur_idx = directions.index(current_direction)
        tgt_idx = directions.index(target_dir)

        diff = (tgt_idx - cur_idx) % 4
        if diff == 0:
            return 0  # Straight
        elif diff == 1:
            return 1  # Turn Right
        elif diff == 3:
            return 2  # Turn Left
        else:
            # 180-degree reversal is illegal in Snake; default to right turn
            return 1


class AStarAgent:
    """
    Pure A* Snake Agent with Safe Movement & Trap Prevention.
    Strategy:
        1. Search shortest path to food via A*.
        2. If path exists and is safe (tail reachable after eating), follow it.
        3. If path is unsafe or unreachable, follow tail-chasing path to survive.
        4. If tail unreachable, pick the legal neighbor with maximum connected free space.
    """

    def __init__(
        self,
        width: int = SCREEN_WIDTH,
        height: int = SCREEN_HEIGHT,
        block_size: int = BLOCK_SIZE,
    ):
        self.finder = PathFinder(width, height, block_size)
        self.last_path: List[Tuple[int, int]] = []

    def get_action(self, game: "SnakeGameAI") -> int:
        """
        Determine next action (0, 1, or 2) using safe A* pathfinding.
        """
        snake_grid = [to_grid(pt, self.finder.block_size) for pt in game.snake]
        head_grid = snake_grid[0]
        food_grid = to_grid(game.food, self.finder.block_size)
        obstacles = set(snake_grid[:-1])

        # 1. Search path to food
        food_path = self.finder.a_star(head_grid, food_grid, obstacles)

        # 2. Check path safety
        if food_path and len(food_path) >= 2:
            if self.finder.is_path_safe(food_path, snake_grid):
                self.last_path = food_path
                return self.finder.grid_step_to_action(
                    head_grid, food_path[1], game.direction
                )

        # 3. Fallback: Tail-chasing survival path
        tail_path = self.finder.find_tail_chasing_path(head_grid, snake_grid)
        if tail_path and len(tail_path) >= 2:
            self.last_path = tail_path
            return self.finder.grid_step_to_action(
                head_grid, tail_path[1], game.direction
            )

        # 4. Emergency: Pick neighbor with maximum accessible free space
        neighbors = self.finder.get_neighbors(head_grid, obstacles)
        if neighbors:
            best_neighbor = max(
                neighbors,
                key=lambda n: self.finder.flood_fill_free_space(n, obstacles),
            )
            self.last_path = [head_grid, best_neighbor]
            return self.finder.grid_step_to_action(
                head_grid, best_neighbor, game.direction
            )

        # No safe move exists
        self.last_path = []
        return 0


class AStarSafetyShield:
    """
    Advanced Multi-Tiered Safety Shield for DQN + A* Hybrid Agent (Phase 4).
    Guarantees:
        1. Immediate collision rejection (walls and snake body).
        2. BFS accessible space check (dead-end trap prevention).
        3. Tail-reachability validation (ensures exit route always exists).
        4. Safe food route fallback (A* to food if safe).
        5. Tail-chasing survival fallback (keeps snake alive until food route opens).
        6. Emergency maximal open-space BFS fallback.
        7. Full telemetry tracking of decisions, overrides, and rejections.
    """

    def __init__(
        self,
        width: Any = SCREEN_WIDTH,
        height: int = SCREEN_HEIGHT,
        block_size: int = BLOCK_SIZE,
        enabled: bool = True,
    ):
        if hasattr(width, 'width') and hasattr(width, 'height'):
            game = width
            width = getattr(game, 'width', SCREEN_WIDTH)
            height = getattr(game, 'height', SCREEN_HEIGHT)
            block_size = getattr(game, 'block_size', BLOCK_SIZE)
        elif hasattr(width, 'w') and hasattr(width, 'h'):
            game = width
            width = getattr(game, 'w', SCREEN_WIDTH)
            height = getattr(game, 'h', SCREEN_HEIGHT)
            block_size = getattr(game, 'block_size', BLOCK_SIZE)

        self.finder = PathFinder(width, height, block_size)
        self.enabled = enabled
        self.reset_telemetry()

    def reset_telemetry(self) -> None:
        """Reset all safety shield decision counters."""
        self.total_decisions = 0
        self.dqn_actions_approved = 0
        self.actions_rejected = 0
        self.fallback_astar_food = 0
        self.fallback_dqn_alt = 0
        self.fallback_tail_chase = 0
        self.fallback_free_space = 0
        self.last_rejection_reason: str = "None"
        self.rejection_reasons = {
            "immediate_collision": 0,
            "trapped_pocket": 0,
            "tail_unreachable": 0,
        }

    reset_stats = reset_telemetry

    def get_telemetry(self) -> Dict[str, Any]:
        """Return comprehensive telemetry metrics for reporting."""
        intervention_rate = (
            (self.actions_rejected / self.total_decisions * 100.0)
            if self.total_decisions > 0
            else 0.0
        )
        return {
            "total_decisions": self.total_decisions,
            "dqn_actions_approved": self.dqn_actions_approved,
            "actions_rejected": self.actions_rejected,
            "intervention_rate_pct": float(np.round(intervention_rate, 2)),
            "fallback_astar_food": self.fallback_astar_food,
            "fallback_dqn_alt": self.fallback_dqn_alt,
            "fallback_tail_chase": self.fallback_tail_chase,
            "fallback_free_space": self.fallback_free_space,
            "last_rejection_reason": self.last_rejection_reason,
            "rejection_reasons": dict(self.rejection_reasons),
        }

    def evaluate_action_safety(
        self, game: Any, action: int
    ) -> Tuple[bool, str, int]:
        """
        Evaluate candidate action (0=Straight, 1=Right, 2=Left).
        Returns:
            (is_safe: bool, reason: str, free_space: int)
        """
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        idx = directions.index(game.direction)
        if action == 1:
            idx = (idx + 1) % 4
        elif action == 2:
            idx = (idx - 1) % 4
        cand_dir = directions[idx]

        hx, hy = game.snake[0]
        bx = game.block_size
        if cand_dir == "UP":
            cand_pt = (hx, hy - bx)
        elif cand_dir == "DOWN":
            cand_pt = (hx, hy + bx)
        elif cand_dir == "LEFT":
            cand_pt = (hx - bx, hy)
        else:
            cand_pt = (hx + bx, hy)

        # 1. Immediate collision check
        if game.check_collision(cand_pt):
            return False, "immediate_collision", 0

        # 2. BFS flood-fill accessible space
        cand_grid = to_grid(cand_pt, bx)
        snake_grid = [to_grid(p, bx) for p in game.snake]
        obstacles = set(snake_grid[:-1])

        free_space = self.finder.flood_fill_free_space(cand_grid, obstacles)
        # Trapped pocket check (if snake len >= 4 and free space strictly less than body)
        if len(game.snake) >= 4 and free_space < len(game.snake):
            return False, "trapped_pocket", free_space

        # 3. Tail reachability validation (evaluated when accessible space is constrained)
        if len(game.snake) >= 4 and free_space < len(game.snake) * 1.5:
            simulated_body = list(snake_grid)
            simulated_body.insert(0, cand_grid)
            if cand_pt != game.food:
                simulated_body.pop()

            sim_head = simulated_body[0]
            sim_tail = simulated_body[-1]
            sim_obstacles = set(simulated_body[:-1])
            tail_path = self.finder.a_star(sim_head, sim_tail, sim_obstacles)

            if tail_path is None:
                return False, "tail_unreachable", free_space

        return True, "safe", free_space

    def filter_action(
        self,
        game: Any,
        dqn_action: int,
        q_values: Optional[List[float]] = None,
    ) -> Tuple[int, bool, str]:
        """
        Screen the proposed DQN action through the multi-layer safety shield.
        Returns:
            (selected_action: int, was_overridden: bool, source: str)
        """
        self.total_decisions += 1
        self.last_rejection_reason = "None"

        if not self.enabled:
            self.dqn_actions_approved += 1
            return dqn_action, False, "dqn"

        is_safe, reason, _ = self.evaluate_action_safety(game, dqn_action)
        if is_safe:
            self.dqn_actions_approved += 1
            self.last_rejection_reason = "None"
            return dqn_action, False, "dqn"

        # DQN action rejected!
        self.actions_rejected += 1
        self.last_rejection_reason = reason
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

        snake_grid = [to_grid(p, game.block_size) for p in game.snake]
        head_grid = snake_grid[0]
        food_grid = to_grid(game.food, game.block_size)
        obstacles = set(snake_grid[:-1])

        # Priority 1: Safe A* path to food
        food_path = self.finder.a_star(head_grid, food_grid, obstacles)
        if food_path and len(food_path) >= 2 and self.finder.is_path_safe(food_path, snake_grid):
            astar_act = self.finder.grid_step_to_action(head_grid, food_path[1], game.direction)
            act_safe, _, _ = self.evaluate_action_safety(game, astar_act)
            if act_safe:
                self.fallback_astar_food += 1
                return astar_act, True, "astar_food"

        # Priority 2: Alternative DQN actions (ranked by Q-value if provided)
        alt_actions = [a for a in range(3) if a != dqn_action]
        if q_values is not None and len(q_values) == 3:
            alt_actions.sort(key=lambda a: q_values[a], reverse=True)

        for alt in alt_actions:
            alt_safe, _, _ = self.evaluate_action_safety(game, alt)
            if alt_safe:
                self.fallback_dqn_alt += 1
                return alt, True, "dqn_safe_alt"

        # Priority 3: Tail-chasing survival path
        tail_path = self.finder.find_tail_chasing_path(head_grid, snake_grid)
        if tail_path and len(tail_path) >= 2:
            tail_act = self.finder.grid_step_to_action(head_grid, tail_path[1], game.direction)
            tail_safe, _, _ = self.evaluate_action_safety(game, tail_act)
            if tail_safe:
                self.fallback_tail_chase += 1
                return tail_act, True, "tail_chase"

        # Priority 4: Emergency Maximal Free Space BFS
        neighbors = self.finder.get_neighbors(head_grid, obstacles)
        if neighbors:
            best_neighbor = max(
                neighbors,
                key=lambda n: self.finder.flood_fill_free_space(n, obstacles),
            )
            space_act = self.finder.grid_step_to_action(head_grid, best_neighbor, game.direction)
            self.fallback_free_space += 1
            return space_act, True, "free_space"

        # Inevitable collision: execute DQN action
        return dqn_action, True, "forced_dqn"


def evaluate_action_safety(
    game: "SnakeGameAI", action: int, finder: Optional[PathFinder] = None
) -> Tuple[bool, int]:
    """
    Backward-compatible wrapper evaluating action safety.
    Returns (is_safe: bool, free_space_count: int).
    """
    shield = AStarSafetyShield(game.width, game.height, game.block_size)
    is_safe, _, space = shield.evaluate_action_safety(game, action)
    return is_safe, space

