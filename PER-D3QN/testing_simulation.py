import random
import timeit
import numpy as np
import traci

PHASE_NS_GREEN   = 0
PHASE_NS_YELLOW  = 1
PHASE_NSL_GREEN  = 2
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN   = 4
PHASE_EW_YELLOW  = 5
PHASE_EWL_GREEN  = 6
PHASE_EWL_YELLOW = 7

class Simulation:
    def __init__(
        self,
        Model,
        TrafficGen,
        sumo_cmd,
        max_steps,
        green_duration,
        yellow_duration,
        num_states,
        num_actions,
        # —— 与训练保持一致的关键开关 ——
        global_seed_offset=1234,
        use_action_mask=True,
        min_green=8,
        max_green=40,
        no_relief_limit=3,
        relief_drop=1,
        state_normalization="z_score",
        # —— 奖励相关（只做评估统计，不反向传播） ——
        waiting_time_penalty_scale=1.0,
        congestion_threshold=15,
        congestion_penalty=-10.0,
        flow_reward_weight=0.05,
        # —— 小技巧：避免死循环与“长期不放行” ——
        tiny_eval_epsilon=0.02,
        relief_check_near_cells=3,
    ):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._max_steps = int(max_steps)
        self._green = int(green_duration)
        self._yellow = int(yellow_duration)
        self._num_states = int(num_states)
        self._num_actions = int(num_actions)

        self._seed_off = int(global_seed_offset)
        self._mask_on = bool(use_action_mask)
        self._min_g = int(min_green)
        self._max_g = int(max_green)
        self._no_relief_limit = int(no_relief_limit)
        self._relief_drop = int(relief_drop)
        self._norm = str(state_normalization).lower()
        self._tiny_eps = float(tiny_eval_epsilon)
        self._near_k = int(relief_check_near_cells)

        self._wt_scale = float(waiting_time_penalty_scale)
        self._cong_th = int(congestion_threshold)
        self._cong_pen = float(congestion_penalty)
        self._flow_w = float(flow_reward_weight)

        # logs
        self._reward_episode = []
        self._queue_len_episode = []
        self._waiting_time_episode = []
        self._cumu_wait = []

        # runtime
        self._step = 0
        self._waiting_times = {}
        self._sum_wait = 0.0

        # 统计
        self._tp_start = 0
        self._tp_end = 0
        self._no_relief_cnt = np.zeros(self._num_actions, dtype=np.int32)

    def run(self, episode: int):
        # 固定种子（可复现）
        seed = (episode or 0) + self._seed_off
        random.seed(seed); np.random.seed(seed)

        # reset
        self._reward_episode = []
        self._queue_len_episode = []
        self._waiting_time_episode = []
        self._cumu_wait.append(0.0)  # 每回合保存一次总延迟
        self._sum_wait = 0.0
        self._waiting_times = {}
        self._no_relief_cnt[:] = 0
        self._step = 0
        self._tp_start = 0
        self._tp_end = 0

        t0 = timeit.default_timer()
        traci_started = False
        try:
            self._TrafficGen.generate_routefile(seed=seed)
            traci.start(self._sumo_cmd); traci_started = True

            old_total_wait = 0.0
            old_action = -1

            while self._step < self._max_steps:
                state = self._get_state()

                # 动作选择：掩码 + “被饿”偏置 + 微探索
                action = self._choose_action(state)

                # 切相：需要黄灯
                if self._step != 0 and action != old_action:
                    self._set_yellow_phase(old_action)
                    self._simulate(self._yellow)

                # 绿灯（夹在最小/最大之间，避免频繁切/长时间不切）
                g = min(max(self._green, self._min_g), self._max_g)
                self._set_green_phase(action)
                self._simulate(g)

                # —— 与训练一致的“评估用”奖励（只统计，不学习） ——
                current_total_wait = self._collect_waiting_times()

                # 1) 等待下降 × 缩放
                wait_delta = (old_total_wait - current_total_wait) if self._step > 0 else 0.0
                base_reward = self._wt_scale * wait_delta

                # 2) 拥堵惩罚（取四个进口的最大排队）
                dir_halts = {
                    'N': traci.edge.getLastStepHaltingNumber("N2TL"),
                    'S': traci.edge.getLastStepHaltingNumber("S2TL"),
                    'E': traci.edge.getLastStepHaltingNumber("E2TL"),
                    'W': traci.edge.getLastStepHaltingNumber("W2TL"),
                }
                max_q = max(dir_halts.values())
                cong_r = self._cong_pen * (max_q / float(self._cong_th)) if max_q > self._cong_th else 0.0

                # 3) 放行量正奖励（轻量近似：队列下降的正部分）
                q_dec = 0.0
                if len(self._queue_len_episode) >= 2:
                    q_dec = max(0.0, self._queue_len_episode[-2] - self._queue_len_episode[-1])
                flow_r = self._flow_w * q_dec

                reward = float(base_reward + cong_r + flow_r)

                # 记账
                old_action = action
                old_total_wait = current_total_wait

                self._reward_episode.append(reward)
                self._waiting_time_episode.append(float(current_total_wait))
                self._sum_wait += float(current_total_wait)

                # 更新“是否缓解”计数
                self._update_relief_counters(action)

            sim_time = round(timeit.default_timer() - t0, 3)

        finally:
            if traci_started:
                traci.close()

        # 保存本回合总延迟
        self._cumu_wait[-1] = float(self._sum_wait)

        return {
            "simulation_time": sim_time if 'sim_time' in locals() else 0.0,
            "total_reward": float(np.sum(self._reward_episode)) if self._reward_episode else 0.0,
            "mean_queue": float(np.mean(self._queue_len_episode)) if self._queue_len_episode else 0.0,
            "max_queue": int(np.max(self._queue_len_episode)) if self._queue_len_episode else 0,
            "sum_wait": float(self._sum_wait),
            "mean_wait": float(np.mean(self._waiting_time_episode)) if self._waiting_time_episode else 0.0,
            "steps": int(self._step),
            "teleport_starts": int(self._tp_start),
            "teleport_ends": int(self._tp_end),
        }

    # --------- 动作选择（掩码 + 被饿偏置 + 微探索） ---------
    def _choose_action(self, state: np.ndarray) -> int:
        q = np.asarray(self._Model.predict_one(state))[0]

        if self._mask_on:
            mask = self._build_action_mask(state)
            q = q.copy()
            q[mask == 0] = -1e9

        # 被饿动作的小幅正偏置
        bias = np.minimum(self._no_relief_cnt.astype(np.float32), self._no_relief_limit) * 0.15
        q = q + bias

        # 微探索，防止陷入循环
        if np.random.rand() < self._tiny_eps:
            valid = np.where(q > -1e8)[0]
            if len(valid) > 0:
                return int(np.random.choice(valid))
        return int(np.argmax(q))

    def _build_action_mask(self, state: np.ndarray) -> np.ndarray:
        mask = np.zeros(self._num_actions, dtype=int)

        def has_nearby(g: int) -> int:
            base = g * 10; k = min(self._near_k, 10)
            return int(np.any(state[base: base + k] > 0.5))

        # 动作到车道组的映射（与你的训练一致）
        mask[0] = max(has_nearby(2), has_nearby(6))  # NS 直
        mask[1] = max(has_nearby(3), has_nearby(7))  # NS 左
        mask[2] = max(has_nearby(4), has_nearby(0))  # EW 直
        mask[3] = max(has_nearby(5), has_nearby(1))  # EW 左
        if mask.sum() == 0:
            mask[:] = 1
        return mask

    def _update_relief_counters(self, action: int):
        # 若队列下降则视为“缓解”，对应动作计数下降；否则累加
        if len(self._queue_len_episode) >= 2 and self._queue_len_episode[-1] <= self._queue_len_episode[-2]:
            self._no_relief_cnt[action] = max(0, self._no_relief_cnt[action] - self._relief_drop)
        else:
            self._no_relief_cnt[action] += 1

    # --------- SUMO 推进/指标 ---------
    def _simulate(self, steps_todo: int):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step
        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            q = self._get_queue_length()
            self._queue_len_episode.append(int(q))
            self._tp_start += traci.simulation.getStartingTeleportNumber()
            self._tp_end   += traci.simulation.getEndingTeleportNumber()

    def _collect_waiting_times(self) -> float:
        incoming = ["E2TL", "N2TL", "W2TL", "S2TL"]
        for vid in traci.vehicle.getIDList():
            wt = traci.vehicle.getAccumulatedWaitingTime(vid)
            edge = traci.vehicle.getRoadID(vid)
            if edge in incoming:
                self._waiting_times[vid] = wt
            else:
                self._waiting_times.pop(vid, None)
        return float(sum(self._waiting_times.values()))

    def _set_yellow_phase(self, old_action: int):
        if old_action is None or old_action < 0: return
        traci.trafficlight.setPhase("TL", int(old_action) * 2 + 1)

    def _set_green_phase(self, a: int):
        if   a == 0: traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif a == 1: traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif a == 2: traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif a == 3: traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)
        else:        traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)

    def _get_queue_length(self) -> int:
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        return int(halt_N + halt_S + halt_E + halt_W)

    def _get_state(self) -> np.ndarray:
        # 按训练的编码：计数→(可选)标准化
        s = np.zeros(self._num_states, dtype=np.float32)
        for vid in traci.vehicle.getIDList():
            lane = traci.vehicle.getLaneID(vid)
            pos_raw = traci.vehicle.getLanePosition(vid)
            d = 750.0 - float(pos_raw)
            if d < 0: continue
            if   d < 7:    cell = 0
            elif d < 14:   cell = 1
            elif d < 21:   cell = 2
            elif d < 28:   cell = 3
            elif d < 40:   cell = 4
            elif d < 60:   cell = 5
            elif d < 100:  cell = 6
            elif d < 160:  cell = 7
            elif d < 400:  cell = 8
            elif d <= 750: cell = 9
            else:          continue

            if   lane in ("W2TL_0","W2TL_1","W2TL_2"): g = 0
            elif lane == "W2TL_3":                      g = 1
            elif lane in ("N2TL_0","N2TL_1","N2TL_2"): g = 2
            elif lane == "N2TL_3":                      g = 3
            elif lane in ("E2TL_0","E2TL_1","E2TL_2"): g = 4
            elif lane == "E2TL_3":                      g = 5
            elif lane in ("S2TL_0","S2TL_1","S2TL_2"): g = 6
            elif lane == "S2TL_3":                      g = 7
            else: continue

            idx = g * 10 + cell
            if 0 <= idx < self._num_states:
                s[idx] += 1.0

        if self._norm == "binary":
            s = (s > 0).astype(np.float32)
        elif self._norm == "z_score":
            m, sd = float(np.mean(s)), float(np.std(s))
            s = (s - m) / sd if sd > 1e-8 else (s - m)
        elif self._norm == "count_normalized":
            m, sd = float(np.mean(s)), float(np.std(s))
            s = (s - m) / sd if sd > 1e-8 else (s - m)
            mn, mx = float(np.min(s)), float(np.max(s))
            s = (s - mn) / (mx - mn) if (mx - mn) > 1e-8 else s
        return s

    # 只读属性
    @property
    def queue_length_episode(self): return self._queue_len_episode
    @property
    def reward_episode(self): return self._reward_episode
    @property
    def cumulative_wait_store(self): return self._cumu_wait
    @property
    def waiting_time_episode(self): return self._waiting_time_episode
    @property
    def sum_waiting_time(self): return self._sum_wait
