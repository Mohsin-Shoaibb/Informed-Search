"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Dynamic Pathfinding Agent — A* & Greedy Best-First Search          ║
║              Redesigned: Clean, Sophisticated, Clutter-Free GUI              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dependencies:  pip install pygame
Run:           python pathfinding_agent.py

Layout:
  ┌─────────────────────────────────────────────────────────┐
  │  TOP BAR  — Title  +  4 live metric pills               │
  ├──────────┬──────────────────────────────────────────────┤
  │          │                                              │
  │  LEFT    │           GRID  (centred)                    │
  │  PANEL   │                                              │
  │  230 px  │                                              │
  │          │                                              │
  └──────────┴──────────────────────────────────────────────┘
"""

import pygame
import heapq
import random
import time
import math

# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
WIN_W, WIN_H    = 1280, 760
TOPBAR_H        = 52
PANEL_W         = 230
FPS             = 60
AGENT_STEP_MS   = 75
DYN_SPAWN_PROB  = 0.028
MIN_CELL, MAX_CELL = 10, 52

# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════
INK     = (  8,  12,  20)
DEEP    = ( 13,  18,  30)
SURFACE = ( 20,  28,  46)
BORDER  = ( 34,  46,  72)
MUTED   = ( 70,  88, 120)
TEXT    = (195, 210, 230)
BRIGHT  = (230, 240, 255)

TEAL   = (  0, 210, 185)
AMBER  = (255, 185,  35)
ROSE   = (255,  72, 120)
SKY    = ( 56, 165, 255)
LIME   = ( 72, 220, 100)
ORANGE = (255, 135,  40)

C_EMPTY    = ( 16,  24,  42)
C_WALL     = ( 38,  52,  82)
C_GRID     = ( 22,  32,  56)
C_START    = TEAL
C_GOAL     = AMBER
C_FRONTIER = (250, 228,  60)
C_VISITED  = ( 48,  96, 220)
C_PATH     = LIME
C_AGENT    = ROSE
C_REPLAN   = ORANGE

EMPTY = 0; WALL = 1; START = 2; GOAL = 3
FRONTIER = 4; VISITED = 5; PATH = 6

CELL_COLORS = {
    EMPTY: C_EMPTY, WALL: C_WALL, START: C_START, GOAL: C_GOAL,
    FRONTIER: C_FRONTIER, VISITED: C_VISITED, PATH: C_PATH,
}

# ══════════════════════════════════════════════════════════════════════════════
#  FONT REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
F = {}

def init_fonts():
    F["title"] = pygame.font.SysFont("segoeui", 17, bold=True)
    F["head"]  = pygame.font.SysFont("segoeui", 13, bold=True)
    F["body"]  = pygame.font.SysFont("segoeui", 13)
    F["small"] = pygame.font.SysFont("segoeui", 11)
    F["pill"]  = pygame.font.SysFont("segoeui", 12, bold=True)
    F["btn"]   = pygame.font.SysFont("segoeui", 13, bold=True)
    F["mono"]  = pygame.font.SysFont("consolas", 12)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def lerp_c(a, b, t):
    return tuple(max(0, min(255, int(a[i] + (b[i]-a[i])*t))) for i in range(3))

def rrect(surf, col, rect, r=6, bw=0, bc=None):
    pygame.draw.rect(surf, col, rect, border_radius=r)
    if bw and bc:
        pygame.draw.rect(surf, bc, rect, bw, border_radius=r)

# ══════════════════════════════════════════════════════════════════════════════
#  PRIORITY QUEUE
# ══════════════════════════════════════════════════════════════════════════════
class PQ:
    def __init__(self):
        self._h = []; self._n = 0
    def push(self, pri, item):
        heapq.heappush(self._h, (pri, self._n, item)); self._n += 1
    def pop(self):
        return heapq.heappop(self._h)[2]
    def __bool__(self): return bool(self._h)
    def __len__(self):  return len(self._h)

# ══════════════════════════════════════════════════════════════════════════════
#  HEURISTICS & SEARCH
# ══════════════════════════════════════════════════════════════════════════════
def manhattan(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
def euclidean(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])

def search(grid, start, goal, algorithm="astar", heuristic="manhattan"):
    rows, cols = len(grid), len(grid[0])
    h_fn = manhattan if heuristic == "manhattan" else euclidean
    t0   = time.perf_counter()

    pq = PQ()
    came_from = {}
    g_cost    = {start: 0.0}
    in_open   = {start}
    pq.push(h_fn(start, goal), start)

    visited_list = []
    front_list   = []
    closed       = set()

    while pq:
        cur = pq.pop()
        if cur in closed: continue
        in_open.discard(cur)
        closed.add(cur)

        visited_list.append(cur)
        front_list.append(frozenset(in_open))

        if cur == goal: break
        r, c = cur
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
                nb = (nr, nc)
                if nb in closed: continue
                ng = g_cost[cur] + 1
                if nb not in g_cost or ng < g_cost[nb]:
                    g_cost[nb]    = ng
                    came_from[nb] = cur
                    hv = h_fn(nb, goal)
                    fv = hv if algorithm == "gbfs" else ng + hv
                    pq.push(fv, nb)
                    in_open.add(nb)

    ms = (time.perf_counter() - t0) * 1000.0
    path = []
    if goal in came_from or goal == start:
        n = goal
        while n in came_from:
            path.append(n); n = came_from[n]
        path.append(start); path.reverse()
    return path, visited_list, front_list, ms

# ══════════════════════════════════════════════════════════════════════════════
#  GRID MODEL
# ══════════════════════════════════════════════════════════════════════════════
class Grid:
    def __init__(self, rows, cols):
        self.rows  = rows; self.cols = cols
        self.cells = [[EMPTY]*cols for _ in range(rows)]
        self.start = (0, 0);           self.goal = (rows-1, cols-1)
        self.cells[0][0]           = START
        self.cells[rows-1][cols-1] = GOAL

    def in_bounds(self, r, c): return 0 <= r < self.rows and 0 <= c < self.cols
    def set_start(self, r, c):
        self.cells[self.start[0]][self.start[1]] = EMPTY
        self.start = (r,c); self.cells[r][c] = START
    def set_goal(self, r, c):
        self.cells[self.goal[0]][self.goal[1]] = EMPTY
        self.goal  = (r,c); self.cells[r][c] = GOAL
    def place_wall(self, r, c):
        if (r,c) not in (self.start, self.goal): self.cells[r][c] = WALL
    def remove_wall(self, r, c):
        if self.cells[r][c] == WALL: self.cells[r][c] = EMPTY
    def clear_overlay(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cells[r][c] in (FRONTIER, VISITED, PATH): self.cells[r][c] = EMPTY
    def clear_walls(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cells[r][c] == WALL: self.cells[r][c] = EMPTY
    def generate(self, density):
        self.clear_overlay()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) == self.start:     self.cells[r][c] = START
                elif (r,c) == self.goal:    self.cells[r][c] = GOAL
                else: self.cells[r][c] = WALL if random.random() < density else EMPTY
    def raw(self):
        return [[WALL if v==WALL else EMPTY for v in row] for row in self.cells]
    def apply_overlay(self, visited, frontier, path):
        self.clear_overlay()
        for (r,c) in visited:
            if (r,c) not in (self.start, self.goal): self.cells[r][c] = VISITED
        for (r,c) in frontier:
            if (r,c) not in (self.start, self.goal): self.cells[r][c] = FRONTIER
        for (r,c) in path:
            if (r,c) not in (self.start, self.goal): self.cells[r][c] = PATH

# ══════════════════════════════════════════════════════════════════════════════
#  BUTTON
# ══════════════════════════════════════════════════════════════════════════════
class Button:
    def __init__(self, rect, label, accent=SKY, active=False, danger=False):
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.accent = (210, 50, 50) if danger else accent
        self.active = active
        self.danger = danger

    def draw(self, surf):
        hov = self.rect.collidepoint(pygame.mouse.get_pos())
        if self.active:
            fill = self.accent; tc = BRIGHT; bc = lerp_c(self.accent, BRIGHT, 0.4)
        elif hov:
            fill = lerp_c(SURFACE, self.accent, 0.22); tc = BRIGHT; bc = self.accent
        else:
            fill = SURFACE; tc = TEXT; bc = BORDER
        rrect(surf, fill, self.rect, r=5, bw=1, bc=bc)
        img = F["btn"].render(self.label, True, tc)
        surf.blit(img, img.get_rect(center=self.rect.center))

    def hit(self, ev):
        return (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and self.rect.collidepoint(ev.pos))

# ══════════════════════════════════════════════════════════════════════════════
#  SLIDER
# ══════════════════════════════════════════════════════════════════════════════
class Slider:
    def __init__(self, label, mn, mx, val, integer=True):
        self.label = label; self.mn = mn; self.mx = mx
        self.value = val;   self.integer = integer
        self.rect  = pygame.Rect(0, 0, 100, 4)
        self._drag = False

    @property
    def val(self): return int(self.value) if self.integer else round(self.value, 2)

    def _kx(self):
        t = (self.value - self.mn) / max(self.mx - self.mn, 1)
        return int(self.rect.x + t * self.rect.w)

    def draw(self, surf, x, y, w):
        self.rect = pygame.Rect(x, y + 14, w, 3)
        kx = self._kx(); ky = self.rect.centery

        # label + value
        li = F["small"].render(self.label, True, MUTED)
        vi = F["pill"].render(str(self.val), True, TEAL)
        surf.blit(li, (x, y))
        surf.blit(vi, (x + w - vi.get_width(), y))

        # track
        pygame.draw.rect(surf, BORDER, self.rect, border_radius=2)
        fr = pygame.Rect(x, y + 14, kx - x, 3)
        pygame.draw.rect(surf, TEAL, fr, border_radius=2)

        # knob
        pygame.draw.circle(surf, BRIGHT, (kx, ky), 7)
        pygame.draw.circle(surf, TEAL,   (kx, ky), 5)

    def handle(self, ev):
        kx = self._kx(); ky = self.rect.centery
        kr = pygame.Rect(kx-9, ky-9, 18, 18)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if kr.collidepoint(ev.pos) or self.rect.collidepoint(ev.pos):
                self._drag = True
        if ev.type == pygame.MOUSEBUTTONUP: self._drag = False
        if ev.type == pygame.MOUSEMOTION and self._drag:
            t = max(0.0, min(1.0, (ev.pos[0]-self.rect.x) / max(self.rect.w,1)))
            raw = self.mn + t*(self.mx-self.mn)
            self.value = int(round(raw)) if self.integer else round(raw, 2)

# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class App:
    S_IDLE = "idle";  S_ANIM = "animating"
    S_WALK = "walking"; S_DYN = "dynamic"

    def __init__(self):
        pygame.init()
        init_fonts()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
        pygame.display.set_caption("Dynamic Pathfinding Agent")
        self.clock  = pygame.time.Clock()

        self.sl_rows  = Slider("Rows",       5, 40,  20)
        self.sl_cols  = Slider("Cols",       5, 55,  25)
        self.sl_dens  = Slider("Density %",  0, 65,  30)
        self.sl_speed = Slider("Anim Speed", 1, 120, 40)
        self.sliders  = [self.sl_rows, self.sl_cols, self.sl_dens, self.sl_speed]

        self.grid      = Grid(self.sl_rows.val, self.sl_cols.val)
        self.algorithm = "astar"
        self.heuristic = "manhattan"
        self.state     = self.S_IDLE

        self.anim_steps   = []; self.anim_idx  = 0
        self.final_path   = []; self.agent_pos = None; self.agent_idx = 0
        self.last_anim_ms = 0;  self.last_step_ms = 0
        self.replan_flash = 0

        self.m_visited = 0; self.m_path = 0
        self.m_exec    = 0.0; self.m_replans = 0; self.found = False

        self.paint = None; self.held = False
        self._build_buttons()

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build_buttons(self):
        px = 12; bw = PANEL_W - 24; bh = 29

        def B(y, lbl, **kw):
            return Button((px, TOPBAR_H + y, bw, bh), lbl, **kw)

        # Algorithm (y=28 → section header at y=16)
        self.btn_astar = B( 28, "A*  Search",       accent=SKY,  active=True)
        self.btn_gbfs  = B( 62, "Greedy BFS",        accent=SKY)

        # Heuristic (section header at y=100)
        self.btn_manh  = B(116, "Manhattan",         accent=TEAL, active=True)
        self.btn_eucl  = B(150, "Euclidean",         accent=TEAL)

        # Edit tools (section header at y=188)
        self.btn_wall  = B(204, "  Draw Walls",     accent=AMBER)
        self.btn_erase = B(238, "  Erase",          accent=AMBER)
        self.btn_start = B(272, "  Set Start",      accent=TEAL)
        self.btn_goal  = B(306, "  Set Goal",       accent=AMBER)

        # Map tools (section header at y=344)
        self.btn_gen   = B(360, "  Generate Map")
        self.btn_clear = B(394, "  Clear Walls")

        # Run (section header at y=432)
        self.btn_search  = B(448, "  Visualise",   accent=LIME)
        self.btn_dynamic = B(482, "  Dynamic",    accent=ORANGE)
        self.btn_stop    = B(516, "  Stop",        danger=True)

        self.btn_all = [
            self.btn_astar, self.btn_gbfs,
            self.btn_manh, self.btn_eucl,
            self.btn_wall, self.btn_erase, self.btn_start, self.btn_goal,
            self.btn_gen, self.btn_clear,
            self.btn_search, self.btn_dynamic, self.btn_stop,
        ]

    # ── Geometry ──────────────────────────────────────────────────────────────
    def _grid_area(self):
        sw, sh = self.screen.get_size()
        return pygame.Rect(PANEL_W, TOPBAR_H, sw - PANEL_W, sh - TOPBAR_H)

    def _cell_size(self):
        ga = self._grid_area()
        cs = min(ga.w // self.grid.cols, ga.h // self.grid.rows)
        return max(MIN_CELL, min(MAX_CELL, cs))

    def _origin(self):
        ga = self._grid_area(); cs = self._cell_size()
        return (ga.x + (ga.w - cs*self.grid.cols)//2,
                ga.y + (ga.h - cs*self.grid.rows)//2)

    def _to_cell(self, px, py):
        ox, oy = self._origin(); cs = self._cell_size()
        c = (px-ox)//cs; r = (py-oy)//cs
        if self.grid.in_bounds(r, c): return r, c
        return None

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw_topbar(self):
        sw, _ = self.screen.get_size()
        surf  = self.screen
        pygame.draw.rect(surf, DEEP, (0, 0, sw, TOPBAR_H))
        pygame.draw.line(surf, BORDER, (0, TOPBAR_H-1), (sw, TOPBAR_H-1), 1)

        # App title
        ti = F["title"].render("DYNAMIC PATHFINDING AGENT", True, BRIGHT)
        surf.blit(ti, (PANEL_W + 16, (TOPBAR_H - ti.get_height())//2))

        # Status indicator
        sc = LIME if self.found else (ROSE if self.m_visited > 0 else MUTED)
        st_map = {True: "Path Found", False: "No Path"}
        st_txt = st_map.get(self.found, "Ready") if self.m_visited > 0 else "Ready"
        pygame.draw.circle(surf, sc, (PANEL_W + 16 + ti.get_width() + 14, TOPBAR_H//2), 5)
        si = F["small"].render(st_txt, True, sc)
        surf.blit(si, (PANEL_W + 16 + ti.get_width() + 24, TOPBAR_H//2 - si.get_height()//2))

        # Metric pills — right side
        pills = [
            ("VISITED",   str(self.m_visited),        SKY),
            ("PATH COST", str(self.m_path),           LIME),
            ("TIME",      f"{self.m_exec:.1f} ms",    TEAL),
            ("RE-PLANS",  str(self.m_replans),        ORANGE),
        ]
        rx = sw - 12
        for lbl_txt, val_txt, col in reversed(pills):
            lw = max(F["small"].size(lbl_txt)[0], F["pill"].size(val_txt)[0])
            pw = lw + 22; ph = 36
            py_p = (TOPBAR_H - ph) // 2
            rx  -= pw + 8
            pr   = pygame.Rect(rx, py_p, pw, ph)
            rrect(surf, SURFACE, pr, r=6, bw=1, bc=BORDER)
            li = F["small"].render(lbl_txt, True, MUTED)
            vi = F["pill"].render(val_txt, True, col)
            surf.blit(li, li.get_rect(centerx=pr.centerx, top=pr.top+4))
            surf.blit(vi, vi.get_rect(centerx=pr.centerx, bottom=pr.bottom-4))

    def _draw_panel(self):
        _, sh = self.screen.get_size()
        surf  = self.screen

        pygame.draw.rect(surf, DEEP, (0, TOPBAR_H, PANEL_W, sh-TOPBAR_H))
        pygame.draw.line(surf, BORDER, (PANEL_W-1, TOPBAR_H), (PANEL_W-1, sh), 1)

        px = 12

        # Helper: section divider + micro-label
        def section(text, y_offset):
            gy = TOPBAR_H + y_offset
            pygame.draw.line(surf, BORDER, (px, gy-2), (PANEL_W-px, gy-2), 1)
            si = F["small"].render(text, True, MUTED)
            surf.blit(si, (px, gy))

        section("ALGORITHM",  16)
        section("HEURISTIC", 100)
        section("EDIT TOOLS", 188)
        section("MAP",        344)
        section("RUN",        432)

        # State badge (top of panel, below topbar)
        state_info = {
            self.S_IDLE: ("● IDLE",      MUTED),
            self.S_ANIM: ("● SEARCHING", SKY),
            self.S_WALK: ("● WALKING",   LIME),
            self.S_DYN:  ("● DYNAMIC",   ORANGE),
        }
        st, sc = state_info.get(self.state, ("—", MUTED))
        si = F["pill"].render(st, True, sc)
        surf.blit(si, (px, TOPBAR_H + 3))

        # Paint mode hint
        if self.paint:
            hints = {"wall": "Click+drag to draw",
                     "erase":"Click+drag to erase",
                     "start":"Click cell → set start",
                     "goal": "Click cell → set goal"}
            hi = F["small"].render(hints.get(self.paint,""), True, AMBER)
            surf.blit(hi, (px, TOPBAR_H + 18))

        # Current config line (above sliders)
        alg  = "A*" if self.algorithm=="astar" else "GBFS"
        heur = "Manh." if self.heuristic=="manhattan" else "Eucl."
        ci = F["small"].render(f"Config:  {alg}  ·  {heur}", True, TEAL)
        surf.blit(ci, (px, sh - 185))

        # Sliders (bottom section)
        for i, sl in enumerate(self.sliders):
            sl.draw(surf, px+2, sh - 150 + i*40, PANEL_W - px*2 - 4)

        # All buttons
        for btn in self.btn_all:
            btn.draw(surf)

    def _draw_grid(self):
        surf = self.screen; cs = self._cell_size(); ox, oy = self._origin()
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                x = ox + c*cs; y = oy + r*cs
                pygame.draw.rect(surf, CELL_COLORS.get(self.grid.cells[r][c], C_EMPTY), (x,y,cs,cs))
                if cs >= 13:
                    pygame.draw.rect(surf, C_GRID, (x,y,cs,cs), 1)

        if self.agent_pos:
            r, c  = self.agent_pos
            cx    = ox + c*cs + cs//2; cy = oy + r*cs + cs//2
            rad   = max(3, cs//2 - 2)
            pygame.draw.circle(surf, C_AGENT, (cx, cy), rad)
            if cs >= 14:
                pygame.draw.circle(surf, BRIGHT, (cx, cy), rad, 1)
            if self.replan_flash > 0:
                t  = self.replan_flash / 24
                rc = lerp_c(SURFACE, C_REPLAN, t)
                pygame.draw.circle(surf, rc, (cx, cy), rad+5, 2)
                self.replan_flash -= 1

    def _draw(self):
        self.screen.fill(INK)
        self._draw_grid()
        self._draw_topbar()
        self._draw_panel()
        pygame.display.flip()

    # ── Search / walk logic ───────────────────────────────────────────────────
    def _run_search(self):
        self.grid.clear_overlay()
        self.state = self.S_ANIM; self.anim_idx = 0
        self.agent_pos = None; self.m_replans = 0
        raw = self.grid.raw()
        path, visited, fronts, ms = search(raw, self.grid.start, self.grid.goal,
                                           self.algorithm, self.heuristic)
        self.anim_steps = list(zip(visited, fronts))
        self.final_path = path
        self.m_visited  = len(visited)
        self.m_path     = len(path)-1 if len(path)>1 else 0
        self.m_exec     = ms; self.found = bool(path)
        self.last_anim_ms = pygame.time.get_ticks()

    def _run_dynamic(self):
        self.grid.clear_overlay(); self.m_replans = 0
        raw = self.grid.raw()
        path, visited, _, ms = search(raw, self.grid.start, self.grid.goal,
                                      self.algorithm, self.heuristic)
        self.final_path = path; self.m_visited = len(visited)
        self.m_path = len(path)-1 if len(path)>1 else 0
        self.m_exec = ms; self.found = bool(path)
        if not path: self.state = self.S_IDLE; return
        self.agent_pos = path[0]; self.agent_idx = 0
        self.state = self.S_DYN; self.last_step_ms = pygame.time.get_ticks()
        self.grid.apply_overlay(set(), set(), path)

    def _step_anim(self):
        now = pygame.time.get_ticks()
        delay = max(1, 122 - self.sl_speed.val)
        if now - self.last_anim_ms < delay: return
        self.last_anim_ms = now
        if self.anim_idx < len(self.anim_steps):
            vis = set(v for v,_ in self.anim_steps[:self.anim_idx+1])
            _, fr = self.anim_steps[self.anim_idx]
            self.grid.apply_overlay(vis, fr, [])
            self.anim_idx += 1
        else:
            self.grid.apply_overlay(set(v for v,_ in self.anim_steps), set(), self.final_path)
            if self.final_path:
                self.agent_pos = self.final_path[0]; self.agent_idx = 0
                self.state = self.S_WALK; self.last_step_ms = pygame.time.get_ticks()
            else:
                self.state = self.S_IDLE

    def _step_walk(self):
        now = pygame.time.get_ticks()
        if now - self.last_step_ms < AGENT_STEP_MS: return
        self.last_step_ms = now
        self.agent_idx += 1
        if self.agent_idx >= len(self.final_path):
            self.agent_pos = self.grid.goal; self.state = self.S_IDLE; return
        self.agent_pos = self.final_path[self.agent_idx]

    def _step_dynamic(self):
        now = pygame.time.get_ticks()
        if now - self.last_step_ms < AGENT_STEP_MS: return
        self.last_step_ms = now
        if random.random() < DYN_SPAWN_PROB:
            r = random.randint(0, self.grid.rows-1)
            c = random.randint(0, self.grid.cols-1)
            if (r,c) not in (self.grid.start, self.grid.goal, self.agent_pos):
                remaining = set(self.final_path[self.agent_idx:])
                self.grid.place_wall(r, c)
                if (r,c) in remaining:
                    self._replan(); return
        self.agent_idx += 1
        if self.agent_idx >= len(self.final_path):
            self.agent_pos = self.grid.goal; self.state = self.S_IDLE; return
        self.agent_pos = self.final_path[self.agent_idx]
        self.grid.apply_overlay(set(), set(), self.final_path[self.agent_idx:])

    def _replan(self):
        self.m_replans += 1; self.replan_flash = 24
        cur = self.agent_pos or self.grid.start
        raw = self.grid.raw()
        path, vis, _, ms = search(raw, cur, self.grid.goal, self.algorithm, self.heuristic)
        self.m_exec += ms; self.m_visited += len(vis)
        if path:
            self.final_path = path; self.agent_idx = 0
            self.found = True; self.m_path = len(path)-1
            self.grid.apply_overlay(set(), set(), path)
        else:
            self.found = False; self.state = self.S_IDLE

    def _stop(self):
        self.state = self.S_IDLE; self.agent_pos = None
        self.final_path = []; self.anim_steps = []
        self.grid.clear_overlay()

    def _rebuild(self):
        self._stop(); self.grid = Grid(self.sl_rows.val, self.sl_cols.val)
        self.m_visited = self.m_path = self.m_replans = 0
        self.m_exec = 0.0; self.found = False

    # ── Events ────────────────────────────────────────────────────────────────
    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: return False
            for sl in self.sliders: sl.handle(ev)

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                cell = self._to_cell(*ev.pos)
                on_grid = cell and ev.pos[0] > PANEL_W and ev.pos[1] > TOPBAR_H
                if on_grid:
                    r, c = cell
                    if self.paint == "wall"  and self.state == self.S_IDLE: self.grid.place_wall(r,c);  self.held = True
                    elif self.paint == "erase" and self.state == self.S_IDLE: self.grid.remove_wall(r,c); self.held = True
                    elif self.paint == "start" and self.state == self.S_IDLE: self.grid.set_start(r,c); self.paint = None
                    elif self.paint == "goal"  and self.state == self.S_IDLE: self.grid.set_goal(r,c);  self.paint = None

            if ev.type == pygame.MOUSEBUTTONUP:   self.held = False
            if ev.type == pygame.MOUSEMOTION and self.held:
                cell = self._to_cell(*ev.pos)
                if cell and ev.pos[0] > PANEL_W and ev.pos[1] > TOPBAR_H:
                    r, c = cell
                    if self.paint == "wall":  self.grid.place_wall(r,c)
                    elif self.paint == "erase": self.grid.remove_wall(r,c)

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if self.btn_gen.hit(ev):    self._rebuild(); self.grid.generate(self.sl_dens.val/100)
                elif self.btn_clear.hit(ev) and self.state==self.S_IDLE:
                    self.grid.clear_walls(); self.grid.clear_overlay()
                elif self.btn_astar.hit(ev):
                    self.algorithm="astar";  self.btn_astar.active=True;  self.btn_gbfs.active=False
                elif self.btn_gbfs.hit(ev):
                    self.algorithm="gbfs";   self.btn_gbfs.active=True;   self.btn_astar.active=False
                elif self.btn_manh.hit(ev):
                    self.heuristic="manhattan"; self.btn_manh.active=True; self.btn_eucl.active=False
                elif self.btn_eucl.hit(ev):
                    self.heuristic="euclidean"; self.btn_eucl.active=True; self.btn_manh.active=False
                elif self.btn_wall.hit(ev):   self.paint = "wall"  if self.paint!="wall"  else None
                elif self.btn_erase.hit(ev):  self.paint = "erase" if self.paint!="erase" else None
                elif self.btn_start.hit(ev):  self.paint = "start" if self.paint!="start" else None
                elif self.btn_goal.hit(ev):   self.paint = "goal"  if self.paint!="goal"  else None
                elif self.btn_search.hit(ev)  and self.state==self.S_IDLE:  self._run_search()
                elif self.btn_dynamic.hit(ev) and self.state in (self.S_IDLE, self.S_WALK): self._run_dynamic()
                elif self.btn_stop.hit(ev):   self._stop()

            if ev.type == pygame.KEYDOWN:
                if   ev.key==pygame.K_r:     self._rebuild(); self.grid.generate(self.sl_dens.val/100)
                elif ev.key==pygame.K_SPACE  and self.state==self.S_IDLE: self._run_search()
                elif ev.key==pygame.K_d      and self.state==self.S_IDLE: self._run_dynamic()
                elif ev.key==pygame.K_ESCAPE: self._stop()
                elif ev.key==pygame.K_c      and self.state==self.S_IDLE:
                    self.grid.clear_walls(); self.grid.clear_overlay()

        self.btn_wall.active  = (self.paint=="wall")
        self.btn_erase.active = (self.paint=="erase")
        self.btn_start.active = (self.paint=="start")
        self.btn_goal.active  = (self.paint=="goal")
        return True

    def run(self):
        running = True
        while running:
            running = self._handle_events()
            if   self.state==self.S_ANIM: self._step_anim()
            elif self.state==self.S_WALK: self._step_walk()
            elif self.state==self.S_DYN:  self._step_dynamic()
            self._draw()
            self.clock.tick(FPS)
        pygame.quit()

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    App().run()