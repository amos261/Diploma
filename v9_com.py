# scheduler_with_wash.py

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
                 planning_horizon: int = 24,
                 wash_duration_slots: int = 0,
                 max_simultaneous_washes: int = 5,
                 default_demand: int = 10000,
                 default_penalty: int = 1,
                 time_limit_seconds: int = 600,
                 num_threads: int = 6,
                 max_chunks_per_line = 4):

        self.QUANTUM = 3600 # сколько секунд в слоте
        
        # Параметры расписания
        self.data_file = data_file
        self.horizon = planning_horizon                        # максимальный доступный нам слот времени
        self.wash_duration = wash_duration_slots               # время в слотах мойки/обслуживания и проч.
        self.max_simultaneous_washes = max_simultaneous_washes # какое максимальное количество обслуживаний можно соверашть в один момент, суть - количество обслуживающих машин
        self.default_demand = default_demand                   # масса продукта, требуемая к производству; базово одинакова для всех, но ниже можно настроить для каждого
        self.default_penalty = default_penalty                 # стоимость/ценность товара; тоже базовая(хотя это имеет мало смысла), тоже можно настроить для каждого
        self.time_limit = time_limit_seconds                   # ограничение сверху на работу солвера
        self.num_threads = num_threads                         # количество потоков при решении\
        self.max_chunks_per_line = max_chunks_per_line         # максимальное количество активных интервалов производства на линии, отображает максимальное количество переключнй
        #self.max_chunks_per_line = ceil(self.horizon/(self.wash_duration + self.no_switch_slots))
        
        # Данные (заполняется load_data)
        self.line_ids = []          # список линий
        self.boiler_ids = []        # список бойлеров(машин)
        self.sku_ids = []           # список продуктов
        self.production_rate = {}   # (линия, продукт) -> скорость производства 
        self.line_boiler_map = {}   # линия -> совместимые с ней бойлеры
        self.boiler_sku_map = {}    # продукт -> бойлеры, способные над ним работать
        self.boilers_required = {}  # продукт -> сколько требуется бойлеров для его производства
        self.no_switch_slots = 0    # количество слотов, в течении которых нельзя переключать производство с одного продукта на другой
        self.deadlines = {}         # продукт -> время после которого его нельзя производить(тоже регулируется пользователем)
        

        # Переменные CP-SAT
        self.model = cp_model.CpModel()
        self.solver = None

        # Моделируемые переменные
        self.prod_start = {}     # (продукт, линия) -> IntVar
        self.prod_duration = {}  # (продукт, линия) -> IntVar
        self.prod_end = {}       # (продукт, линия) -> IntVar
        self.prod_active = {}    # (продукт, линия) -> BoolVar
        self.prod_interval = {}  # (продукт, линия) -> OptionalIntervalVar

        self.boiler_intervals = []  # (intervalVar, boiler, sku, line, assignBool)
        self.wash_intervals = []    # (intervalVar, kind, resource, switchBool, startVar)

        self.production_tasks = []  # (продукт, линия)
        self.max_prod_time = {}     # (продукт, линия) -> int
        self.boiler_act        = {}
        self.boiler_start      = {}
        self.boiler_end        = {}

        # Целевая функция
        self.total_produced = {}  # sku -> IntVar
        self.shortfall  = {}      # sku -> IntVar
        self.objective_val  = None


        self.feasible_skus  = []   # Отфильтрованные SKU

        

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
        self.boiler_sku_map  = raw["prodB"]
        self.boilers_required= raw["needB"]
        self.no_switch_slots = raw["noSwitch"]

        # Дедлайны по умолчанию = горизонт
        for sku in self.sku_ids:
            self.deadlines[sku] = self.horizon

        # Рисуем граф линия - продукт
        G = nx.Graph()
        G.add_nodes_from(self.line_ids, bipartite=0)
        G.add_nodes_from(self.sku_ids, bipartite=1)
        for line in self.line_ids:
            for sku in self.sku_ids:
                if (line,sku) in self.production_rate and \
                   len(self.line_boiler_map[line] & self.boiler_sku_map[sku]) >= max(1, self.boilers_required[sku]):
                    G.add_edge(line, sku)
        pos = nx.bipartite_layout(G, self.sku_ids)
        plt.figure(figsize=(8,6))
        nx.draw_networkx_nodes(G, pos, nodelist=self.line_ids,  node_color="lightblue", label="Lines")
        nx.draw_networkx_nodes(G, pos, nodelist=self.sku_ids,   node_color="salmon",    label="Skus")
        nx.draw_networkx_labels(G, pos, font_size=9)
        nx.draw_networkx_edges(G, pos, edge_color="gray")
        plt.axis("off")
        plt.legend(scatterpoints=1)
        plt.title("Совместимость линии и продукта")
        plt.show()

        #print(f"[Load]  Данные и граф за {time.time()-t0:.2f}s")

    def filter_skus(self):
        # Оставляем только технически выполнимые SKU
        t0 = time.time()
        self.feasible_skus = []
        for sku in self.sku_ids:
            for line in self.line_ids:
                if (line,sku) not in self.production_rate:
                    continue
                boilers = [b for b in self.boiler_ids
                           if b in self.line_boiler_map[line]
                           and b in self.boiler_sku_map[sku]]
                if len(boilers) >= max(1, self.boilers_required[sku]):
                    self.feasible_skus.append(sku)
                    break
        print(f"[Filter] {len(self.feasible_skus)}/{len(self.sku_ids)} очистка невозможных к производству SKU. ") #за {time.time()-t0:.2f}s


    def build_model(self):
        m = cp_model.CpModel()


        H         = self.horizon
        lines     = self.line_ids
        boilers   = self.boiler_ids
        skus      = self.feasible_skus
        speed     = self.production_rate
        lb_map    = self.line_boiler_map
        bs_map    = self.boiler_sku_map
        needB     = self.boilers_required
        no_sw     = self.no_switch_slots
        wash_sl   = self.wash_duration
        demand    = self.default_demand
        penalty   = self.default_penalty
        max_chunks= self.max_chunks_per_line
        ddl       = self.deadlines

        # 1) Создаём интервалы для каждой пары (линия, чанк, продукт)
        self.iv_lkp = {}
        for l in lines:
            for k in range(max_chunks):
                for p in skus:
                    # Если линия не может делать этот продукт пропускаем
                    if speed.get((l,p), 0) == 0:
                        continue
                    # Если один минимальный чанк сам больше дедлайна, ничего не влезает
                    if no_sw > ddl[p]:
                        continue
                    # хоть что-то произвели
                    if no_sw * speed[(l,p)] < 1:
                        continue

                    # запускаем ли p в чанке (l,k)
                    x  = m.NewBoolVar(f"x_l{l}_k{k}_p{p}")
                    # Время старта, длительность и конец
                    st = m.NewIntVar(0, H, f"st_l{l}_k{k}_p{p}")
                    dr = m.NewIntVar(0, H, f"dr_l{l}_k{k}_p{p}")
                    en = m.NewIntVar(0, H, f"en_l{l}_k{k}_p{p}")
                    # Опциональный интервал: если x=1, то [st,st+dr) реально используется
                    iv = m.NewOptionalIntervalVar(st, dr, en, x, f"iv_l{l}_k{k}_p{p}")

                    # Если x=0, то длительность=0
                    m.Add(dr == 0).OnlyEnforceIf(x.Not())
                    # Если x=1, длительность ≥ min_chunk
                    m.Add(dr >= no_sw).OnlyEnforceIf(x)
                    # Если x=1, конец = старт + длительность
                    m.Add(en == st + dr).OnlyEnforceIf(x)

                    self.iv_lkp[(l, k, p)] = (iv, st, dr, en, x)

        # 2) запрещаем пустой чанк перед активным чанком,
        #    чтобы избежать эквивалентных расписаний, сдвинутых влево без смысла.
        for l in lines:
            for k in range(1, max_chunks):
                prev_xs = [
                    self.iv_lkp[(l, k - 1, p)][4]
                    for p in skus if (l, k - 1, p) in self.iv_lkp
                ]
                curr_xs = [
                    self.iv_lkp[(l, k, p)][4]
                    for p in skus if (l, k, p) in self.iv_lkp
                ]
                if prev_xs and curr_xs:
                    m.Add(sum(prev_xs) >= sum(curr_xs))

        # 3) В одном чанке (l,k) производится не более одного продукта p
        for l in lines:
            for k in range(max_chunks):
                xs = [
                    self.iv_lkp[(l, k, p)][4]
                    for p in skus if (l, k, p) in self.iv_lkp
                ]
                if xs:
                    m.Add(sum(xs) <= 1)

        # 4) Интервалы непересекаются
        for l in lines:
            ivs = [
                iv for (ll, _, _), (iv, *_ ) in self.iv_lkp.items()
                if ll == l
            ]
            if ivs:
                m.AddNoOverlap(ivs)

        # 5) Назначаем бойлеры к запущенным интервалам
        self.ivb = {}
        for (l, k, p), (iv, st, dr, en, x) in self.iv_lkp.items():
             #  Собираем список всех бойлеров, которые подходят и линии, и продукту
            bos = [
                b for b in lb_map[l]
                if b in bs_map.get(p, [])
            ]
            if not bos:
                continue

            picks = []
            needed = max(1, needB.get(p, 1))
            for b in bos:
                y   = m.NewBoolVar(f"y_b{b}_l{l}_k{k}_p{p}")  # бойлер b участвует в этом интервале ровно с теми же st, dr, en, что и линия
                ivb = m.NewOptionalIntervalVar(st, dr, en, y,
                                               f"ivb_{b}_l{l}_k{k}_p{p}")
                m.Add(y <= x) #Если бойлер участвует (y=1), то соответствующий чанк на линии тоже должен быть активен (x=1)
                self.ivb[(b, l, k, p)] = (ivb, y)
                picks.append(y)
            #Если чанк x=1 (линия выпускает p), то ровно needed переменных y должны быть=1;
            # если x=0 (линия простаивает), то все y=0 (ни один бойлер не занят)
            m.Add(sum(picks) == needed).OnlyEnforceIf(x)
            m.Add(sum(picks) == 0).OnlyEnforceIf(x.Not())

        # 6) Непересечение интервалов одного бойлера
        for b in boilers:
            ivs_b = [
                ivb for (bb, _, _, _), (ivb, _) in self.ivb.items()
                if bb == b
            ]
            if ivs_b:
                m.AddNoOverlap(ivs_b)

        # 7) Мойки на линиях
        #   Проходим по всем парам интервалов на одной и той же линии l,
        # в разных чанках k1 < k2, и вставляем мойку, если меняется продукт.
        wash_intervals = []
        for l in lines:
            
            
            for k1 in range(max_chunks):#  Перебираем первый чанк (k1, p1)
                for p1 in skus:
                    key1 = (l, k1, p1)
                    if key1 not in self.iv_lkp:  # Если чанк (l, k1) не может содержать продукт p1, пропускаем
                        continue
                    iv1, st1, dr1, en1, x1 = self.iv_lkp[key1]

                    for k2 in range(k1 + 1, max_chunks):  # Перебираем второй чанк (k2, p2) с k2 > k1
                        for p2 in skus:
                            if p2 == p1: # можно отказаться
                                continue
                            key2 = (l, k2, p2)
                            if key2 not in self.iv_lkp: # Если чанк (l, k2) не может проиизв продукт p2, пропускаем
                                continue
                            iv2, st2, dr2, en2, x2 = self.iv_lkp[key2]

                            # Проверяем, нет ли активных чанков между k1 и k2


                            if k2 - k1 > 1:
                                mids = [
                                    self.iv_lkp[(l, m, pp)][4]
                                    for m in range(k1 + 1, k2)
                                    for pp in skus
                                    if (l, m, pp) in self.iv_lkp
                                ]
                                nb = m.NewBoolVar(f"no_act_l{l}_k{k1}_to_{k2}")
                                if mids:
                                    m.AddBoolAnd([mid.Not() for mid in mids]).OnlyEnforceIf(nb)
                                    m.AddBoolOr(mids).OnlyEnforceIf(nb.Not())
                                else:
                                    m.Add(nb == 1)
                            else:
                                nb = m.NewConstant(1)

                            both_prod = m.NewBoolVar(          # both_prod = оба  включены
                                f"both_l{l}_k{k1}_{p1}_and_k{k2}_{p2}"
                            )
                            m.AddBoolAnd([x1, x2]).OnlyEnforceIf(both_prod)
                            m.AddBoolOr([x1.Not(), x2.Not()]).OnlyEnforceIf(both_prod.Not())

                            both_adj = m.NewBoolVar( # both_adj = оба запущены и между ними нет активных чанков»
                                f"both_adj_l{l}_k{k1}_{p1}_k{k2}_{p2}"
                            )

                            # Дальше создаем интервал мойки...
                            m.AddBoolAnd([both_prod, nb]).OnlyEnforceIf(both_adj)
                            m.AddBoolOr([both_prod.Not(), nb.Not()]).OnlyEnforceIf(both_adj.Not())

                            order = m.NewBoolVar(
                                f"order_l{l}_{k1}_{p1}_to_{k2}_{p2}"
                            )
                            stw = m.NewIntVar(
                                0, H, f"stw_line_{l}_{k1}_{p1}_to_{k2}_{p2}"
                            )
                            enw = m.NewIntVar(
                                0, H, f"enw_line_{l}_{k1}_{p1}_to_{k2}_{p2}"
                            )
                            ivw = m.NewOptionalIntervalVar(
                                stw, wash_sl, enw, both_adj,
                                f"wash_line_{l}_{k1}_{p1}_{k2}_{p2}"
                            )
                            m.Add(enw == stw + wash_sl).OnlyEnforceIf(both_adj)

                            #p1 -> мойка -> p2

                            m.Add(iv1.EndExpr() + wash_sl <= iv2.StartExpr()
                                  ).OnlyEnforceIf([both_adj, order])
                            m.Add(stw >= iv1.EndExpr()
                                  ).OnlyEnforceIf([both_adj, order])
                            m.Add(enw <= iv2.StartExpr()
                                  ).OnlyEnforceIf([both_adj, order])


                            #p2 -> мойка -> p1бб

                            m.Add(iv2.EndExpr() + wash_sl <= iv1.StartExpr()
                                  ).OnlyEnforceIf([both_adj, order.Not()])
                            m.Add(stw >= iv2.EndExpr()
                                  ).OnlyEnforceIf([both_adj, order.Not()])
                            m.Add(enw <= iv1.StartExpr()
                                  ).OnlyEnforceIf([both_adj, order.Not()])

                            wash_intervals.append(ivw)

        # 8) Мойки на бойлерах 
        for b in boilers:
            prods = [
                (ln, ch, pr, ivb, y)
                for (bb, ln, ch, pr), (ivb, y) in self.ivb.items()
                if bb == b
            ]
            for i in range(len(prods)):
                l1, k1, p1, ivb1, y1 = prods[i]
                for j in range(i + 1, len(prods)):
                    l2, k2, p2, ivb2, y2 = prods[j]
                    if p2 == p1:
                        continue

                    seq_min = no_sw + wash_sl + no_sw
                    cant_12 = (seq_min > ddl[p2])
                    cant_21 = (seq_min > ddl[p1])
                    if cant_12 and cant_21:
                        m.Add(y1 + y2 <= 1)
                        continue

                    both_prod_b = m.NewBoolVar(
                        f"bothb_{b}_{l1}_{k1}_{p1}_to_{l2}_{k2}_{p2}"
                    )
                    m.AddBoolAnd([y1, y2]).OnlyEnforceIf(both_prod_b)
                    m.AddBoolOr([y1.Not(), y2.Not()]).OnlyEnforceIf(both_prod_b.Not())

                    orderb = m.NewBoolVar(
                        f"order_b{b}_{l1}_{k1}_{p1}_to_{l2}_{k2}_{p2}"
                    )
                    stwb = m.NewIntVar(
                        0, H,
                        f"stw_boiler_{b}_{l1}_{k1}_{p1}_{l2}_{k2}_{p2}"
                    )
                    enwb = m.NewIntVar(
                        0, H,
                        f"enw_boiler_{b}_{l1}_{k1}_{p1}_{l2}_{k2}_{p2}"
                    )
                    ivwb = m.NewOptionalIntervalVar(
                        stwb, wash_sl, enwb, both_prod_b,
                        f"wash_boiler_{b}_{l1}_{k1}_{p1}_to_{l2}_{k2}_{p2}"
                    )
                    m.Add(enwb == stwb + wash_sl).OnlyEnforceIf(both_prod_b)

                    m.Add(ivb1.EndExpr() + wash_sl <= ivb2.StartExpr()
                          ).OnlyEnforceIf([both_prod_b, orderb])
                    m.Add(stwb >= ivb1.EndExpr()
                          ).OnlyEnforceIf([both_prod_b, orderb])
                    m.Add(enwb <= ivb2.StartExpr()
                          ).OnlyEnforceIf([both_prod_b, orderb])

                    m.Add(ivb2.EndExpr() + wash_sl <= ivb1.StartExpr()
                          ).OnlyEnforceIf([both_prod_b, orderb.Not()])
                    m.Add(stwb >= ivb2.EndExpr()
                          ).OnlyEnforceIf([both_prod_b, orderb.Not()])
                    m.Add(enwb <= ivb1.StartExpr()
                          ).OnlyEnforceIf([both_prod_b, orderb.Not()])

                    wash_intervals.append(ivwb)

        # 9) Ограничение на одновременную мойку
        if wash_intervals:
            m.AddCumulative(wash_intervals, [1] * len(wash_intervals), self.max_simultaneous_washes)

        # 10) Целевая функция
        terms = []
        for p in skus:
            vols = []
            for (l, k, pp), (iv, st, dr, en, run_var) in self.iv_lkp.items():
                if pp != p:
                    continue
                vols.append(dr * speed[(l, p)])
            prod  = m.NewIntVar(0, demand, f"prod_{p}")
            short = m.NewIntVar(0, demand, f"short_{p}")
            m.Add(prod == sum(vols))
            m.Add(short >= demand - prod)
            terms.append(short * penalty)
        m.Minimize(sum(terms))

        self.model = m

        print("[Build] Модель построена.")






    def solve(self):
        t0 = time.time()
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = self.time_limit
        self.solver.parameters.num_search_workers   = self.num_threads
        status = self.solver.Solve(self.model)
        self.objective_val = self.solver.ObjectiveValue()
        print(f"[Solve]  Status: {self.solver.StatusName(status)}, "
              f"Objective={self.objective_val:.0f}, "
              f"time={time.time()-t0:.2f}s")

    def plot_gantt(self):
        import matplotlib.pyplot as plt
        import itertools

        solver = self.solver
        H      = self.horizon
        wash   = self.wash_duration    # ← вот здесь: правильно берем длительность мойки

        # Вертикальные позиции
        y_line   = {l: i for i, l in enumerate(self.line_ids)}
        y_boiler = {b: i for i, b in enumerate(self.boiler_ids)}

        fig, (axL, axB) = plt.subplots(2, 1, sharex=True, figsize=(16, 10))
        palette   = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        color_map = {}

        def draw_block(ax, y, start, dur, sku):
            c = color_map.setdefault(sku, next(palette))
            ax.barh(y, dur, left=start, height=0.6, color=c, edgecolor="black")
            ax.text(start + dur/2, y, str(sku),
                    ha="center", va="center", color="white", fontsize=8)

        # 1) Production on Lines + washes
        for l in self.line_ids:
            tasks = []
            for (ll, k, p), (iv, _, _, _, x) in self.iv_lkp.items():
                if ll == l and solver.Value(x):
                    s = solver.Value(iv.StartExpr())
                    d = solver.Value(iv.SizeExpr())
                    tasks.append((s, d, p))
            tasks.sort(key=lambda t: t[0])
            for s, d, p in tasks:
                draw_block(axL, y_line[l], s, d, p)
            for (s1, d1, _), (s2, d2, _) in zip(tasks, tasks[1:]):
                gap = s2 - (s1 + d1)
                if gap >= wash:
                    axL.barh(y_line[l], wash,
                             left=s1 + d1,
                             height=0.3,
                             color="gray", alpha=0.5)

        axL.set_yticks(list(y_line.values()))
        axL.set_yticklabels(self.line_ids)
        axL.set_xlim(0, H)
        axL.set_title("Диаграмма Ганта производства по линиям, серый - мойка")

        # 2) Production on Boilers + washes
        for b in self.boiler_ids:
            tasks = []
            for (bb, l, k, p), (ivb, y) in self.ivb.items():
                if bb == b and solver.Value(y):
                    s = solver.Value(ivb.StartExpr())
                    d = solver.Value(ivb.SizeExpr())
                    tasks.append((s, d, p))
            tasks.sort(key=lambda t: t[0])
            for s, d, p in tasks:
                draw_block(axB, y_boiler[b], s, d, p)
            for (s1, d1, _), (s2, d2, _) in zip(tasks, tasks[1:]):
                gap = s2 - (s1 + d1)
                if gap >= wash:
                    axB.barh(y_boiler[b], wash,
                             left=s1 + d1,
                             height=0.3,
                             color="gray", alpha=0.5)

        axB.set_yticks(list(y_boiler.values()))
        axB.set_yticklabels(self.boiler_ids)
        axB.set_xlim(0, H)
        axB.set_xlabel("Слоты времени")
        axB.set_title("Диаграмма Ганта производства по бойлерам")

        # 3) Легенда по SKU
        handles, labels = axL.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axL.legend(by_label.values(), by_label.keys(), loc="upper right")

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

if __name__=="__main__":
    main()
