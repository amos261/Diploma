:- discontiguous switch_time/3.
:- discontiguous switch_time/2.
:- discontiguous boiler_produces/2.
:- discontiguous connected/2.
:- discontiguous group_by_sku / 2.
:- discontiguous sku_restriction / 3.
:- discontiguous sku_required_boilers_num / 2.

line('Line A'). % Предикат задаёт имена линий
line('Line 1').
line('Line 2').
line('Line 3').
line('Line 4').
line('Line B').
line('Line C').

boiler('Boiler 1').  % Предикат задаёт имена бойлеров
boiler('Boiler 2').
boiler('Boiler 3').
boiler('Boiler 4').
boiler('Boiler 5').
boiler('Boiler 6').
boiler('Boiler 7').

big('Boiler 1'). % Предикат задаёт, что бойлер B является большим; остальные маленькие
big('Boiler 5').

sku_id(204). % Предикат задаёт имена продуктов; SKU — stock keeping unit
sku_id(101).
sku_id(102).
sku_id(103).
sku_id(104).
sku_id(105).
sku_id(106).
sku_id(107).
sku_id(108).
sku_id(106).
sku_id(201).
sku_id(202).
sku_id(203).
sku_id(107).
sku_id(205).
sku_id(206).
sku_id(207).
sku_id(208).
sku_id(209).
sku_id(210).
sku_id(211).
sku_id(212).
sku_id(213).


boiler_produces('Boiler 1', 101). % Бойлер может производить продукт
boiler_produces('Boiler 1', 102).
boiler_produces('Boiler 1', 103).
boiler_produces('Boiler 1', 104).
boiler_produces('Boiler 1', 105).
boiler_produces('Boiler 1', 106).
boiler_produces('Boiler 1', 107).
boiler_produces('Boiler 1', 108).
boiler_produces('Boiler 1', 106).
boiler_produces('Boiler 1', 201).
boiler_produces('Boiler 1', 202).
boiler_produces('Boiler 1', 203).
boiler_produces('Boiler 1', 205).
boiler_produces('Boiler 1', 206).
boiler_produces('Boiler 1', 207).
boiler_produces('Boiler 1', 208).
boiler_produces('Boiler 1', 209).
boiler_produces('Boiler 1', 210).
boiler_produces('Boiler 1', 211).
boiler_produces('Boiler 1', 212).
boiler_produces('Boiler 1', 213).
boiler_produces('Boiler 2', 101).
boiler_produces('Boiler 2', 102).
boiler_produces('Boiler 2', 103).
boiler_produces('Boiler 2', 104).
boiler_produces('Boiler 2', 105).
boiler_produces('Boiler 2', 106).
boiler_produces('Boiler 2', 107).
boiler_produces('Boiler 2', 108).
boiler_produces('Boiler 2', 106).
boiler_produces('Boiler 3', 101).
boiler_produces('Boiler 3', 102).
boiler_produces('Boiler 3', 103).
boiler_produces('Boiler 3', 104).
boiler_produces('Boiler 3', 105).
boiler_produces('Boiler 3', 106).
boiler_produces('Boiler 3', 107).
boiler_produces('Boiler 3', 108).
boiler_produces('Boiler 3', 106).
boiler_produces('Boiler 4', 101).
boiler_produces('Boiler 4', 102).
boiler_produces('Boiler 4', 103).
boiler_produces('Boiler 4', 104).
boiler_produces('Boiler 4', 105).
boiler_produces('Boiler 4', 106).
boiler_produces('Boiler 4', 107).
boiler_produces('Boiler 4', 108).
boiler_produces('Boiler 4', 106).
boiler_produces('Boiler 5', 101).
boiler_produces('Boiler 5', 102).
boiler_produces('Boiler 5', 103).
boiler_produces('Boiler 5', 104).
boiler_produces('Boiler 5', 105).
boiler_produces('Boiler 5', 106).
boiler_produces('Boiler 5', 107).
boiler_produces('Boiler 5', 108).
boiler_produces('Boiler 5', 106).
boiler_produces('Boiler 5', 107).
cleans('Контур 1', 'Line 1', 4200.0). % Автомойщик Контур k моет машину M за T секунд
cleans('Контур 1', 'Line 2', 5400.0).
cleans('Контур 1', 'Line 3', 5400.0).
cleans('Контур 1', 'Line 4', 5400.0).
cleans('Контур 2', 'Boiler 1', 6000.0).
cleans('Контур 2', 'Boiler 2', 6000.0).
cleans('Контур 2', 'Boiler 3', 6000.0).
cleans('Контур 2', 'Boiler 4', 6000.0).
cleans('Контур 2', 'Boiler 5', 6000.0).
cleans('Контур 3', 'Boiler 1', 6000.0).
cleans('Контур 3', 'Boiler 2', 6000.0).
cleans('Контур 3', 'Boiler 3', 6000.0).
cleans('Контур 3', 'Boiler 4', 6000.0).
cleans('Контур 3', 'Boiler 5', 6000.0).
cleans('Контур 5', 'Line A', 7800.0).
cleans('Контур 5', 'Boiler 6', 7200.0).
cleans('Контур 5', 'Boiler 7', 7200.0).
cleans('Контур 5', 'Boiler 8', 7200.0).
cleans('Контур 5', 'Line B', 5400.0).
cleans('Контур 5', 'Line C', 5400.0).

connected('Line A', 'Boiler 6'). % Предикат задаёт связи между линиями и бойлерами
connected('Line A', 'Boiler 7').
connected('Line A', 'Boiler 1').
connected('Line 1', 'Boiler 8').
connected('Line 1', 'Boiler 1').
connected('Line 1', 'Boiler 2').
connected('Line 1', 'Boiler 3').
connected('Line 1', 'Boiler 4').
connected('Line 1', 'Boiler 5').
connected('Line 2', 'Boiler 8').
connected('Line 2', 'Boiler 1').
connected('Line 2', 'Boiler 2').
connected('Line 2', 'Boiler 3').
connected('Line 2', 'Boiler 4').
connected('Line 2', 'Boiler 5').
connected('Line 3', 'Boiler 8').
connected('Line 3', 'Boiler 1').
connected('Line 3', 'Boiler 2').
connected('Line 3', 'Boiler 3').
connected('Line 3', 'Boiler 4').
connected('Line 3', 'Boiler 5').
connected('Line 4', 'Boiler 8').
connected('Line 4', 'Boiler 1').
connected('Line 4', 'Boiler 2').
connected('Line 4', 'Boiler 3').
connected('Line 4', 'Boiler 4').
connected('Line 4', 'Boiler 5').
connected('Line B', 'Boiler 6').
connected('Line B', 'Boiler 7').
connected('Line C', 'Boiler 6').
connected('Line C', 'Boiler 7').
disinfection_time_('Line A', 1800.0). % Время дезинфекции машины
disinfection_time_('Line 1', 1800.0).
disinfection_time_('Line 2', 1800.0).
disinfection_time_('Line 3', 1800.0).
disinfection_time_('Boiler 6', 1800.0).
disinfection_time_('Boiler 7', 1500.0).
disinfection_time_('Boiler 1', 1800.0).
disinfection_time_('Boiler 2', 1800.0).
disinfection_time_('Boiler 3', 1800.0).
disinfection_time_('Boiler 4', 1800.0).
disinfection_time_('Boiler 5', 1800.0).
disinfection_time_('Line 4', 1800.0).
disinfection_time_('Line B', 1800.0).
disinfection_time_('Line C', 1800.0).
manual_cleaning_time_('Line A', 3600).
manual_cleaning_time_('Line 1', 1200).
manual_cleaning_time_('Line 2', 3000).
manual_cleaning_time_('Line 3', 3000).
manual_cleaning_time_('Line 4', 3000).
manual_cleaning_time_('Boiler 1', 1800).
manual_cleaning_time_('Boiler 2', 1800).
manual_cleaning_time_('Boiler 3', 1800).
manual_cleaning_time_('Boiler 4', 1800).
manual_cleaning_time_('Boiler 5', 1800).
manual_cleaning_time_('Boiler 6', 3600).
manual_cleaning_time_('Boiler 7', 3600).
manual_cleaning_time_('Boiler 8', 3600).
manual_cleaning_time_('Line B', 7200).
manual_cleaning_time_('Line C', 7200).
needs_big_boiler(101). % Для производства SKU нужен большой бойлер
needs_big_boiler(102).
needs_big_boiler(103).
needs_big_boiler(104).
needs_big_boiler(105).
needs_big_boiler(106).
needs_big_boiler(107).

no_switch_time(10800). % Время с момента начала производства SKU, до истечения которого запрещено переключение на другой SKU

performance('Line 1', 106, 334.504). % Производительность: сколько кг./час производит линия
performance('Line 1', 107, 249.481).
performance('Line 1', 108, 308.596).
performance('Line 2', 101, 385.129).
performance('Line 2', 102, 270.096).
performance('Line 2', 103, 376.999).
performance('Line 2', 104, 321.426).
performance('Line 2', 105, 367.257).
performance('Line 2', 106, 369.252).
performance('Line 2', 107, 305.275).
performance('Line 2', 108, 405.762).
performance('Line 2', 106, 422.027).
performance('Line 3', 101, 353.301).
performance('Line 3', 102, 352.987).
performance('Line 3', 103, 390.307).
performance('Line 3', 104, 317.283).
performance('Line 3', 105, 380.968).
performance('Line 3', 106, 391.458).
performance('Line 3', 107, 241.57).
performance('Line 3', 108, 366.717).
performance('Line 3', 106, 272.067).
performance('Line 4', 101, 478.358).
performance('Line 4', 102, 431.319).
performance('Line 4', 103, 399.314).
performance('Line 4', 104, 552.722).
performance('Line 4', 105, 437.901).
performance('Line 4', 106, 543.378).
performance('Line 4', 107, 422.3).
performance('Line 4', 108, 472.494).
performance('Line 4', 106, 430.984).
performance('Line A', 201, 650.044).
performance('Line A', 202, 556.648).
performance('Line A', 203, 592.375).
performance('Line B', 204, 1469.714).
performance('Line B', 205, 1239.466).
performance('Line B', 206, 1358.113).
performance('Line B', 207, 1233.932).
performance('Line B', 208, 1025.0).
performance('Line B', 209, 1079.592).
performance('Line B', 210, 1148.017).
performance('Line B', 211, 1055.323).
performance('Line B', 212, 1363.319).
performance('Line B', 213, 1222.305).
performance('Line C', 204, 1492.116).
performance('Line C', 205, 1141.403).
performance('Line C', 206, 1531.657).
performance('Line C', 207, 968.592).
performance('Line C', 208, 1579.74).
performance('Line C', 209, 1163.41).
performance('Line C', 210, 1532.856).
performance('Line C', 211, 1288.668).
performance('Line C', 212, 1224.965).
performance('Line C', 213, 796.575).

production_cycle_times(204, 28800.0, 129600.0, 129600.0). % Время производственного цикла у SKU S: минимальное, оптимальное, максимальное
production_cycle_times(101, 28800.0, 129600.0, 129600.0).
production_cycle_times(102, 28800.0, 129600.0, 129600.0).
production_cycle_times(103, 28800.0, 129600.0, 129600.0).
production_cycle_times(104, 28800.0, 129600.0, 129600.0).
production_cycle_times(105, 28800.0, 129600.0, 129600.0).
production_cycle_times(106, 28800.0, 129600.0, 129600.0).
production_cycle_times(107, 28800.0, 129600.0, 129600.0).
production_cycle_times(108, 28800.0, 129600.0, 129600.0).
production_cycle_times(106, 28800.0, 129600.0, 129600.0).
production_cycle_times(201, 28800.0, 129600.0, 129600.0).
production_cycle_times(202, 28800.0, 129600.0, 129600.0).
production_cycle_times(203, 28800.0, 129600.0, 129600.0).
production_cycle_times(107, 28800.0, 129600.0, 129600.0).
production_cycle_times(205, 28800.0, 129600.0, 129600.0).
production_cycle_times(206, 28800.0, 129600.0, 129600.0).
production_cycle_times(207, 28800.0, 129600.0, 129600.0).
production_cycle_times(208, 28800.0, 129600.0, 129600.0).
production_cycle_times(209, 28800.0, 129600.0, 129600.0).
production_cycle_times(210, 28800.0, 129600.0, 129600.0).
production_cycle_times(211, 28800.0, 129600.0, 129600.0).
production_cycle_times(212, 28800.0, 129600.0, 129600.0).
production_cycle_times(213, 28800.0, 129600.0, 129600.0).


sku_max_lines_num(204, 1). % Максимальное допустимое число работающих линий для производства SKU
sku_max_lines_num(101, 10).
sku_max_lines_num(102, 10).
sku_max_lines_num(103, 10).
sku_max_lines_num(104, 10).
sku_max_lines_num(105, 10).
sku_max_lines_num(106, 4).
sku_max_lines_num(107, 4).
sku_max_lines_num(108, 4).
sku_max_lines_num(106, 10).
sku_max_lines_num(201, 1).
sku_max_lines_num(202, 1).
sku_max_lines_num(203, 1).
sku_max_lines_num(107, 1).
sku_max_lines_num(205, 1).
sku_max_lines_num(206, 1).
sku_max_lines_num(207, 1).
sku_max_lines_num(208, 1).
sku_max_lines_num(209, 1).
sku_max_lines_num(210, 1).
sku_max_lines_num(211, 1).
sku_max_lines_num(212, 1).
sku_max_lines_num(213, 1).
sku_min_lines_num(204, 1). % Минимальное допустимое число работающих линий для производства SKU
sku_min_lines_num(101, 2).
sku_min_lines_num(102, 2).
sku_min_lines_num(103, 2).
sku_min_lines_num(104, 2).
sku_min_lines_num(105, 2).
sku_min_lines_num(106, 1).
sku_min_lines_num(107, 1).
sku_min_lines_num(108, 1).
sku_min_lines_num(106, 2).
sku_min_lines_num(201, 1).
sku_min_lines_num(202, 1).
sku_min_lines_num(203, 1).
sku_min_lines_num(107, 1).
sku_min_lines_num(205, 1).
sku_min_lines_num(206, 1).
sku_min_lines_num(207, 1).
sku_min_lines_num(208, 1).
sku_min_lines_num(209, 1).
sku_min_lines_num(210, 1).
sku_min_lines_num(211, 1).
sku_min_lines_num(212, 1).
sku_min_lines_num(213, 1).

sku_required_boilers_num(204, 1). % Сколько нужно бойлеров для производства SKU
sku_required_boilers_num(101, 3).
sku_required_boilers_num(102, 4).
sku_required_boilers_num(103, 2).
sku_required_boilers_num(104, 3).
sku_required_boilers_num(105, 3).
sku_required_boilers_num(106, 1).
sku_required_boilers_num(107, 1).
sku_required_boilers_num(108, 1).
sku_required_boilers_num(106, 2).
sku_required_boilers_num(201, 1).
sku_required_boilers_num(202, 1).
sku_required_boilers_num(203, 1).
sku_required_boilers_num(107, 2).
sku_required_boilers_num(205, 1).
sku_required_boilers_num(206, 1).
sku_required_boilers_num(207, 1).
sku_required_boilers_num(208, 1).
sku_required_boilers_num(209, 1).
sku_required_boilers_num(210, 1).
sku_required_boilers_num(211, 1).
sku_required_boilers_num(212, 1).
sku_required_boilers_num(213, 1).

switch_time(204, 206, 600.0). % Время переключения машины с одного SKU на другой
switch_time(204, 206, 600.0).
switch_time(204, 206, 600.0).
switch_time(204, 208, 600.0).
switch_time(204, 208, 600.0).
switch_time(204, 208, 600.0).
switch_time(204, 212, 600.0).
switch_time(204, 212, 600.0).
switch_time(204, 212, 600.0).
switch_time(101, 105, 600.0).
switch_time(101, 105, 600.0).
switch_time(101, 105, 600.0).
switch_time(106, 101, 600.0).
switch_time(106, 101, 600.0).
switch_time(106, 101, 600.0).
switch_time(106, 102, 600.0).
switch_time(106, 102, 600.0).
switch_time(106, 102, 600.0).
switch_time(106, 103, 600.0).
switch_time(106, 103, 600.0).
switch_time(106, 103, 600.0).
switch_time(106, 104, 600.0).
switch_time(106, 104, 600.0).
switch_time(106, 104, 600.0).
switch_time(106, 105, 600.0).
switch_time(106, 105, 600.0).
switch_time(106, 105, 600.0).
switch_time(106, 107, 600.0).
switch_time(106, 107, 600.0).
switch_time(106, 107, 600.0).
switch_time(106, 107, 600.0).
switch_time(106, 108, 600.0).
switch_time(106, 108, 600.0).
switch_time(106, 108, 600.0).
switch_time(106, 108, 600.0).
switch_time(106, 106, 600.0).
switch_time(106, 106, 600.0).
switch_time(106, 106, 600.0).
switch_time(107, 104, 600.0).
switch_time(107, 104, 600.0).
switch_time(107, 104, 600.0).
switch_time(106, 104, 600.0).
switch_time(106, 104, 600.0).
switch_time(106, 104, 600.0).
switch_time(201, 202, 600.0).
switch_time(201, 203, 600.0).
switch_time(205, 209, 600.0).
switch_time(205, 209, 600.0).
switch_time(205, 209, 600.0).
switch_time(205, 211, 600.0).
switch_time(205, 211, 600.0).
switch_time(205, 211, 600.0).
switch_time(205, 213, 600.0).
switch_time(205, 213, 600.0).
switch_time(205, 213, 600.0).
switch_time(206, 204, 600.0).
switch_time(206, 204, 600.0).
switch_time(206, 204, 600.0).
switch_time(206, 210, 600.0).
switch_time(206, 210, 600.0).
switch_time(206, 210, 600.0).
switch_time(206, 212, 600.0).
switch_time(206, 212, 600.0).
switch_time(206, 212, 600.0).
switch_time(209, 211, 600.0).
switch_time(209, 211, 600.0).
switch_time(209, 211, 600.0).
switch_time(209, 213, 600.0).
switch_time(209, 213, 600.0).
switch_time(209, 213, 600.0).
switch_time(211, 207, 600.0).
switch_time(211, 207, 600.0).
switch_time(211, 207, 600.0).
switch_time(211, 209, 600.0).
switch_time(211, 209, 600.0).
switch_time(211, 209, 600.0).
switch_time(212, 208, 600.0).
switch_time(212, 208, 600.0).
switch_time(212, 208, 600.0).
