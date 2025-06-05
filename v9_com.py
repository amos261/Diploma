# scheduler_with_wash_fixed.py

import re
import collections
import itertools
import traceback
import time
import math
from math import ceil
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ortools.sat.python import cp_model


def extract_args(line: str):
    """Возвращает список кортежей (str_val, num_val) для всех аргументов."""
    return re.findall(r"'([^']*)'|([-+]?[0-9]*\.?[0-9]+)", line)


class ProductionScheduler:
    def __init__(self,
                 data_file: str,
                 planning_horizon: int = 50,
                 max_chunks_per_line = 3,
                 wash_duration_slots: int = 1,
                 min_chunk_slots: int = 1,
                 max_chunk_slots: int = 4,
                 default_demand: int = 10000,
                 default_penalty: int = 1,
                 time_limit_seconds: int = 120,
                 num_threads: int = 6):

        self.QUANTUM = 3600  # сколько секунд в слоте

        # Параметры расписания
        self.data_file = data_file
        self.horizon = planning_horizon                        # максимальный доступный нам слот времени
        self.default_wash_duration = wash_duration_slots       # базовая длина мойки между парой продуктов
        self.default_demand = default_demand                   # масса продукта, требуемая к производству
        self.default_penalty = default_penalty                 # стоимость/ценность товара
        self.time_limit = time_limit_seconds                   # ограничение сверху на работу солвера
        self.num_threads = num_threads                         # количество потоков при решении
        self.max_chunks_per_line = max_chunks_per_line         # максимальное кол-во активных интервалов на линии

        # Данные (заполняется load_data)
        self.line_ids = []          # список линий
        self.boiler_ids = []        # список бойлеров(машин)
        self.sku_ids = []           # список продуктов
        self.production_rate = {}   # (линия, продукт) -> скорость производства 
        self.line_boiler_map = {}   # линия -> совместимые с ней бойлеры
        self.boiler_sku_map = {}    # продукт -> бойлеры, способные над ним работать
        self.boilers_required = {}  # продукт -> сколько требуется бойлеров для его производства
        self.wash_duration_dict = {}# (некорректо, всегда должно быть постоянным в данной модели)

        self.deadlines = {}         # продукт -> время после которого его нельзя производить

        # Переменные CP-SAT
        self.model = cp_model.CpModel()
        self.solver = None

        # Моделируемые переменные
        self.prod_start = {}     # (продукт, линия) -> IntVar
        self.prod_duration = {}  # (продукт, линия) -> IntVar
        self.prod_end = {}       # (продукт, линия) -> IntVar
        self.prod_active = {}    # (продукт, линия) -> BoolVar
        self.prod_interval = {}  # (продукт, линия) -> OptionalIntervalVar

        self.boiler_intervals = []
        self.wash_intervals = []

        self.min_chunk_length = {}
        self.max_chunk_length = {}
        self.default_min_chunk = min_chunk_slots
        self.default_max_chunk = max_chunk_slots

        self.production_tasks = []  # (продукт, линия)
        self.max_prod_time = {}     # (продукт, линия) -> int
        self.boiler_act = {}
        self.boiler_start = {}
        self.boiler_end = {}

        # Целевая функция
        self.total_produced = {}  # sku -> IntVar
        self.shortfall = {}       # sku -> IntVar
        self.objective_val = None

        self.feasible_skus = []   # Отфильтрованные SKU

    def load_data(self):
        """Загружает syntetic_data.pl, парсит все параметры и рисует bipartite-graph."""
        t0 = time.time()
        raw = collections.defaultdict(dict)
        raw.update({
            "lines": set(), "boilers": set(), "skus": set(),
            "speed": {}, "conn": collections.defaultdict(set),
            "prodB": collections.defaultdict(set),
            "needB": collections.defaultdict(int),
            "noSwitch": 0
        })

        with open(self.data_file, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln.startswith("line("):
                    raw["lines"].add(extract_args(ln)[0][0])
                elif ln.startswith("boiler("):
                    raw["boilers"].add(extract_args(ln)[0][0])
                elif ln.startswith("sku_id("):
                    raw["skus"].add(int(float(extract_args(ln)[0][1])))
                elif ln.startswith(("line_speed(", "performance(")):
                    a = extract_args(ln)
                    line, sku, rate = a[0][0], int(float(a[1][1])), float(a[2][1])
                    raw["speed"][(line, sku)] = ceil(rate * self.QUANTUM / 3600)
                elif ln.startswith("connected("):
                    a = extract_args(ln)
                    raw["conn"][a[0][0]].add(a[1][0])
                elif ln.startswith("boiler_produces("):
                    a = extract_args(ln)
                    raw["prodB"][int(float(a[1][1]))].add(a[0][0])
                elif ln.startswith("sku_required_boilers_num("):
                    a = extract_args(ln)
                    raw["needB"][int(float(a[0][1]))] = int(float(a[1][1]))
                elif ln.startswith("no_switch_time("):
                    sec = float(extract_args(ln)[0][1])
                    raw["noSwitch"] = ceil(sec / self.QUANTUM)

        # Оставляем только линии, для которых есть скорость
        raw["lines"] &= {ln for (ln, _) in raw["speed"]}

        # Копируем в объекты
        self.line_ids = sorted(raw["lines"])
        self.boiler_ids = sorted(raw["boilers"])
        self.sku_ids = sorted(raw["skus"])
        self.production_rate = raw["speed"]
        self.line_boiler_map = raw["conn"]
        self.boiler_sku_map = raw["prodB"]
        self.boilers_required = raw["needB"]
        self.no_switch_slots = raw["noSwitch"]

        for sku in self.sku_ids:
            self.wash_duration_dict[sku] = self.default_wash_duration
            self.min_chunk_length[sku] = self.default_min_chunk
            self.max_chunk_length[sku] = self.default_max_chunk

        # Дедлайны по умолчанию = горизонт
        for sku in self.sku_ids:
            self.deadlines[sku] = self.horizon

        # Рисуем граф линия - продукт
        G = nx.Graph()
        G.add_nodes_from(self.line_ids, bipartite=0)
        G.add_nodes_from(self.sku_ids, bipartite=1)
        for line in self.line_ids:
            for sku in self.sku_ids:
                if (line, sku) in self.production_rate and \
                   len(self.line_boiler_map[line] & self.boiler_sku_map[sku]) >= max(1, self.boilers_required[sku]):
                    G.add_edge(line, sku)
        pos = nx.bipartite_layout(G, self.sku_ids)
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G, pos, nodelist=self.line_ids, node_color="lightblue", label="Lines")
        nx.draw_networkx_nodes(G, pos, nodelist=self.sku_ids, node_color="salmon", label="Skus")
        nx.draw_networkx_labels(G, pos, font_size=9)
        nx.draw_networkx_edges(G, pos, edge_color="gray")
        plt.axis("off")
        plt.legend(scatterpoints=1)
        plt.title("Совместимость линии и продукта")
        plt.show()

        # print(f"[Load]  Данные и граф за {time.time()-t0:.2f}s")

    def filter_skus(self):
        self.feasible_skus = []
        for sku in self.sku_ids:
            for line in self.line_ids:
                if (line, sku) not in self.production_rate:
                    continue
                boilers = [
                    b for b in self.boiler_ids
                    if b in self.line_boiler_map[line] and b in self.boiler_sku_map.get(sku, set())
                ]
                if len(boilers) >= max(1, self.boilers_required.get(sku, 1)):
                    self.feasible_skus.append(sku)
                    break
        print(f"[Filter] {len(self.feasible_skus)}/{len(self.sku_ids)} очищено невозможных к производству SKU.")

    def build_model(self):
        m = cp_model.CpModel()

        # Параметры и словари из объекта
        H = self.horizon
        L = self.line_ids
        B = self.boiler_ids
        P = self.feasible_skus
        v = self.production_rate        # {(l, p): скорость}
        Bl = self.line_boiler_map       # {l: [доступные бойлеры]}
        Bp = self.boiler_sku_map        # {p: {доступные бойлеры}}
        Rp_map = self.boilers_required  # {p: R_p}
        min_chunk = self.default_min_chunk      # delta
        max_chunk = self.default_max_chunk      # gamma
        wash_dur = self.default_wash_duration   # w
        d_map = self.deadlines                  # {p: дедлайн}
        K = self.max_chunks_per_line
        demand = self.default_demand            # n_p
        cost = self.default_penalty             # c_p

        # 1) Создаём переменные и интервал для каждого потенциального «чанка» (l, k, p)
        self.iv_lkp = {}
        for l in L:
            for k in range(K):
                for p in P:
                    if v.get((l, p), 0) == 0:
                        continue
                    x = m.NewBoolVar(f"x_l{l}_k{k}_p{p}")
                    st = m.NewIntVar(0, H, f"st_l{l}_k{k}_p{p}")
                    dr = m.NewIntVar(0, H, f"dr_l{l}_k{k}_p{p}")
                    en = m.NewIntVar(0, H, f"en_l{l}_k{k}_p{p}")
                    iv = m.NewOptionalIntervalVar(st, dr, en, x, f"iv_l{l}_k{k}_p{p}")

                    # Если x=0, интервал нулевой: dr=0 и en=st
                    m.Add(dr == 0).OnlyEnforceIf(x.Not())
                    m.Add(en == st).OnlyEnforceIf(x.Not())

                    # Если x=1,  δ <= dr <= γ,  en = st + dr,  en <= дедлайн[p]
                    m.Add(dr >= min_chunk).OnlyEnforceIf(x)
                    m.Add(dr <= max_chunk).OnlyEnforceIf(x)
                    m.Add(en == st + dr).OnlyEnforceIf(x)
                    m.Add(en <= d_map[p]).OnlyEnforceIf(x)

                    self.iv_lkp[(l, k, p)] = (x, st, dr, en, iv)

        # 2)Sum_p x_{l,p,k} <= 1  — не более одного активного продукта в каждом (l, k)
        for l in L:
            for k in range(K):
                xs = []
                for p in P:
                    key = (l, k, p)
                    if key in self.iv_lkp:
                        xs.append(self.iv_lkp[key][0])
                if xs:
                    m.Add(sum(xs) <= 1)

        # 3) Упорядоченность “чанков” по индексу k: Sum_p x_{l,p,k-1} >= Sum_p x_{l,p,k}
        for l in L:
            for k in range(1, K):
                prev_vars = []
                curr_vars = []
                for p in P:
                    key_prev = (l, k - 1, p)
                    key_curr = (l, k, p)
                    if key_prev in self.iv_lkp:
                        prev_vars.append(self.iv_lkp[key_prev][0])
                    if key_curr in self.iv_lkp:
                        curr_vars.append(self.iv_lkp[key_curr][0])
                if prev_vars and curr_vars:
                    m.Add(sum(prev_vars) >= sum(curr_vars))

        # 4) Временной порядок соседних “чанков” на одной линии.

        for l in L:
            for k in range(K - 1):
                for p1 in P:
                    key1 = (l, k, p1)
                    if key1 not in self.iv_lkp:
                        continue
                    x1, st1, dr1, en1, iv1 = self.iv_lkp[key1]
                    k2 = k + 1
                    for p2 in P:
                        key2 = (l, k2, p2)
                        if key2 not in self.iv_lkp:
                            continue
                        x2, st2, dr2, en2, iv2 = self.iv_lkp[key2]
                        # Теперь независимо от того, p1 == p2 или нет, 
                        # между концом первого и началом второго вставляем мойку
                        m.Add(en1 + wash_dur <= st2).OnlyEnforceIf([x1, x2])

        # 5) Назначение бойлеров: y_{b,l,k,p} и ivb для каждого кандидата из доступных бойлеров
        self.ivb = {}
        for (l, k, p), (x, st, dr, en, iv) in self.iv_lkp.items():
            candidates = [b for b in Bl[l] if b in Bp.get(p, set())]
            if not candidates:
                continue
            R_p = max(1, Rp_map.get(p, 1))
            picks = []
            for b in candidates:
                y = m.NewBoolVar(f"y_b{b}_l{l}_k{k}_p{p}")
                ivb = m.NewOptionalIntervalVar(st, dr, en, y, f"ivb_b{b}_l{l}_k{k}_p{p}")
                m.Add(y <= x)
                self.ivb[(b, l, k, p)] = (y, ivb)
                picks.append(y)
            # Если x=1, ровно R_p бойлеров; если x=0, ни один
            m.Add(sum(picks) == R_p).OnlyEnforceIf(x)
            m.Add(sum(picks) == 0).OnlyEnforceIf(x.Not())

        # 5.1) NoOverlap на интервалах каждого бойлера
        for b in B:
            ivs_b = []
            for (bb, l, k, p), (y, ivb) in self.ivb.items():
                if bb == b:
                    ivs_b.append(ivb)
            if ivs_b:
                m.AddNoOverlap(ivs_b)

        # 5.2) Мойка на бойлерах (параллельные интервалы):
        #      для каждой неупорядоченной пары (i,j)
        #      вводим порядок через булев z и оба варианта порядка через \beta.
        for b in B:
            items = [(y, ivb, p) for (bb, l, k, p), (y, ivb) in self.ivb.items() if bb == b]
            n_items = len(items)
            if n_items < 2:
                continue
            for i in range(n_items):
                y1, ivb1, p1 = items[i]
                start1 = ivb1.StartExpr()
                end1 = ivb1.EndExpr()
                for j in range(i + 1, n_items):
                    y2, ivb2, p2 = items[j]
                    start2 = ivb2.StartExpr()
                    end2 = ivb2.EndExpr()
                    # Всегда ставим мойку wash_dur между концами независимо от  соседства
                    zB = m.NewBoolVar(f"zB_b{b}_i{i}_j{j}_p{p1}_p{p2}")
                    # zB = 1 \Leftrightarrow оба этих интервала реально активны одновременно
                    m.AddBoolAnd([y1, y2]).OnlyEnforceIf(zB)
                    m.AddBoolOr([y1.Not(), y2.Not()]).OnlyEnforceIf(zB.Not())
                    beta = m.NewBoolVar(f"beta_b{b}_i{i}_j{j}_p{p1}_p{p2}")
                    # Первый приходит раньше -> end1 + wash_dur <= start2
                    m.Add(end1 + wash_dur <= start2).OnlyEnforceIf([zB, beta])
                    # Второй приходит раньше -> end2 + wash_dur <= start1
                    m.Add(end2 + wash_dur <= start1).OnlyEnforceIf([zB, beta.Not()])


        # 6) Целевая функция
        terms = []
        self.q_vars = {}
        self.short_vars = {}

        # Предварительно считаем макс. возможный объём для каждого p
        max_volumes = {}
        for p in P:
            total = 0
            for (l, k, pp) in self.iv_lkp:
                if pp == p:
                    total += v.get((l, p), 0) * max_chunk
            max_volumes[p] = total if total > 0 else 0

        for p in P:
            # 6.1) Собираем объёмы = rate * (en - st) для всех (l,k,p)
            volumes_for_p = []
            for (l, k, pp), (x, st, dr, en, iv) in self.iv_lkp.items():
                if pp != p:
                    continue
                rate = v.get((l, p), 0)
                # en - st = dr при x=1, иначе 0
                volumes_for_p.append(rate * (en - st))

            # 6.2) q_p 
            q_p = m.NewIntVar(0, max_volumes[p], f"q_{p}")
            m.Add(q_p == sum(volumes_for_p))

            # 6.3) short_p 
            short_p = m.NewIntVar(0, demand, f"short_{p}")
            m.Add(short_p >= demand - q_p)

            self.q_vars[p] = q_p
            self.short_vars[p] = short_p

            # 6.4) Добавляем штраф cost * short_p
            terms.append(short_p * cost)

        print(f"[Build]  Модель построена")
        m.Minimize(sum(terms))
        self.model = m
        

    def solve(self):
        t0 = time.time()
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = self.time_limit
        self.solver.parameters.num_search_workers = self.num_threads
        status = self.solver.Solve(self.model)
        self.objective_val = self.solver.ObjectiveValue()
        print(f"[Solve]  Status: {self.solver.StatusName(status)}, "
              f"Objective={self.objective_val:.0f}, "
              f"time={time.time()-t0:.2f}s")

    def plot_gantt(self):
        solver = self.solver
        H = self.horizon
        wash = self.default_wash_duration  # длительность мойки

        # Вертикальные позиции для линий и бойлеров
        y_line = {l: i for i, l in enumerate(self.line_ids)}
        y_boiler = {b: i for i, b in enumerate(self.boiler_ids)}

        fig, (axL, axB) = plt.subplots(2, 1, sharex=True, figsize=(16, 10))
        palette = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        color_map = {}

        def draw_block(ax, y, start, dur, sku):
            c = color_map.setdefault(sku, next(palette))
            ax.barh(y, dur, left=start, height=0.6, color=c, edgecolor="black")
            ax.text(start + dur / 2, y, str(sku),
                    ha="center", va="center", color="white", fontsize=8)

        # 1) Производство на линиях с мойками
        for l in self.line_ids:
            tasks = []
            for (ll, k, p), tup in self.iv_lkp.items():
                x, st, dr, en, iv = tup
                if ll == l and solver.Value(x):
                    s = solver.Value(st)
                    d = solver.Value(dr)
                    tasks.append((s, d, p))
            tasks.sort(key=lambda t: t[0])

            # Рисуем продукцию на линиях
            for s, d, p in tasks:
                draw_block(axL, y_line[l], s, d, p)

            # Рисуем мойку (серый прямоугольник) когда gap >= wash
            if len(tasks) > 1:
                for (s1, d1, _), (s2, d2, _) in zip(tasks, tasks[1:]):
                    gap = s2 - (s1 + d1)
                    if gap >= wash:
                        axL.barh(
                            y_line[l],
                            wash,
                            left=s1 + d1,
                            height=0.3,
                            color="gray",
                            alpha=0.5
                        )

        axL.set_yticks(list(y_line.values()))
        axL.set_yticklabels(self.line_ids)
        axL.set_xlim(0, H)
        axL.set_title("Диаграмма Ганта: линии (серый = мойка)")

        # 2) Производство на бойлерах с мойками
        for b in self.boiler_ids:
            tasks = []
            for (bb, l, k, p), (y, ivb) in self.ivb.items():
                if bb == b and solver.Value(y):
                    s = solver.Value(ivb.StartExpr())
                    d = solver.Value(ivb.SizeExpr())
                    tasks.append((s, d, p))
            tasks.sort(key=lambda t: t[0])

            # Рисуем продукцию на бойлерах
            for s, d, p in tasks:
                draw_block(axB, y_boiler[b], s, d, p)

            # Рисуем мойку (серый прямоугольник) когда gap >= wash
            if len(tasks) > 1:
                for (s1, d1, _), (s2, d2, _) in zip(tasks, tasks[1:]):
                    gap = s2 - (s1 + d1)
                    if gap >= wash:
                        axB.barh(
                            y_boiler[b],
                            wash,
                            left=s1 + d1,
                            height=0.3,
                            color="gray",
                            alpha=0.5
                        )

        axB.set_yticks(list(y_boiler.values()))
        axB.set_yticklabels(self.boiler_ids)
        axB.set_xlim(0, H)
        axB.set_xlabel("Слоты времени")
        axB.set_title("Диаграмма Ганта: бойлеры (серый = мойка)")

        plt.tight_layout()
        plt.show()


def main():
    try:
        sched = ProductionScheduler("syntetic_data.pl")
        sched.load_data()
        sched.filter_skus()
        sched.build_model()
        sched.solve()
        sched.plot_gantt()
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
