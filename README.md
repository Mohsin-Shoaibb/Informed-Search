# Dynamic Pathfinding Agent

A real-time pathfinding visualiser built with **Python + Pygame** implementing **A\* Search** and **Greedy Best-First Search** with dynamic obstacle re-planning.

---

## Features

| Category | Details |
|---|---|
| **Algorithms** | A\* Search, Greedy Best-First Search (GBFS) |
| **Heuristics** | Manhattan Distance, Euclidean Distance |
| **Visualisation** | Yellow = Frontier, Blue = Visited, Green = Path, Pink = Agent |
| **Dynamic Mode** | Obstacles spawn randomly while agent walks; auto re-plans if path blocked |
| **Interactive Editor** | Click to draw/erase walls, relocate Start & Goal |
| **Metrics** | Nodes Visited, Path Cost, Exec Time (ms), Re-plan Count |

---

## Requirements

```
Python >= 3.8
pygame >= 2.0
```

---

## Installation

```bash
pip install pygame
```

---

## Run

```bash
python pathfinding_agent.py
```

---

## Controls

### Buttons (Sidebar)
| Button | Action |
|---|---|
| ⟳ Generate Map | Create random maze with chosen density |
| ✕ Clear Walls | Remove all walls |
| A\* / Greedy BFS | Switch algorithm |
| Manhattan / Euclidean | Switch heuristic |
| ✏ Draw Walls | Click/drag on grid to add walls |
| ◻ Erase Walls | Click/drag to remove walls |
| ◎ Set Start | Click a cell to move Start |
| ★ Set Goal | Click a cell to move Goal |
| ▶ Visualise Search | Animate the search exploration + path |
| ⚡ Dynamic Mode | Agent walks; walls appear randomly; auto re-plans |
| ■ Stop | Halt any running animation |

### Keyboard Shortcuts
| Key | Action |
|---|---|
| `Space` | Start visualisation |
| `D` | Start dynamic mode |
| `R` | Regenerate map |
| `Esc` | Stop |

### Sliders
| Slider | Description |
|---|---|
| Rows / Cols | Grid dimensions (requires regenerate) |
| Density % | Wall density for random generation |
| Anim Speed | Controls exploration animation speed |

---

## Algorithm Details

### A\* Search
Uses `f(n) = g(n) + h(n)` where `g(n)` is the cost from start and `h(n)` is the heuristic. **Optimal** — guarantees the shortest path when the heuristic is admissible.

### Greedy Best-First Search
Uses `f(n) = h(n)` only. **Fast but not optimal** — may find suboptimal paths; explores fewer nodes in open environments.

---

## Dynamic Re-planning Logic

1. While the agent traverses the path, new walls spawn at random grid cells with probability `3%` per step.
2. The agent checks whether the new wall intersects its **remaining path**.
3. If yes → immediately re-runs the chosen search algorithm from the agent's **current position**.
4. If no → the agent continues without interruption (efficient; no unnecessary re-computation).

---

## Project Structure

```
pathfinding_agent.py   ← Single-file complete implementation
README.md              ← This file
```
