import numpy as np
import timeit
import traci

# phase codes based on environment.net.xml
PHASE_NS_GREEN   = 0  # action 0
PHASE_NS_YELLOW  = 1
PHASE_NSL_GREEN  = 2  # action 1
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN   = 4  # action 2
PHASE_EW_YELLOW  = 5
PHASE_EWL_GREEN  = 6  # action 3
PHASE_EWL_YELLOW = 7


class D3QNSimulation:
    def __init__(
        self, Model, Memory, TrafficGen, sumo_cmd,
        gamma, max_steps, green_duration, yellow_duration,
        num_states, num_actions, training_epochs,
        warmup_min_samples=None,
        congestion_threshold: int = 20,           # 拥堵阈值（单向排队长度）
        congestion_penalty: float = -5.0,         # 拥堵惩罚值（按比例放大）
        waiting_time_penalty_scale: float = 0.8,  # 等待时间惩罚缩放系数
        state_normalization: str = "z_score",      # 状态归一化：binary, count, count_normalized, z_score
        # ===== 约束参数（防“守绿卡死”） =====
        max_green: int = 40,                      # 最长绿灯（秒）：到时禁止继续同相位
        imbalance_ratio: float = 1.5,             # NS/EW 失衡阈值：一侧>另一侧*ratio 需优先服务
        max_wait_threshold: float = 60.0,         # 任一方向车辆最大累计等待阈值（秒）
        # ===== 放行量正奖励（新增） =====
        flow_reward_weight: float = 0.05,         # 放行量奖励权重（建议 0.02~0.08 微调）
    ):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd

        self._gamma = float(gamma)
        self._max_steps = int(max_steps)
        self._green = int(green_duration)
        self._yellow = int(yellow_duration)
        self._num_states = int(num_states)
        self._num_actions = int(num_actions)
        self._epochs = int(training_epochs)

        # 奖励相关
        self._congestion_threshold = int(congestion_threshold)
        self._congestion_penalty = float(congestion_penalty)
        self._waiting_time_penalty_scale = float(waiting_time_penalty_scale)
        self._flow_w = float(flow_reward_weight)   # ★ 新增：放行量奖励权重

        # 状态归一化
        self._state_normalization = str(state_normalization)

        # 约束
        self._max_green = int(max_green)
        self._imbalance_ratio = float(imbalance_ratio)
        self._max_wait_threshold = float(max_wait_threshold)

        # logs
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._episode_rewards = []
        self._episode_losses = []
        self._congestion_penalties = []

        # runtime
        self._step = 0
        self._warmup_min = warmup_min_samples
        self._current_action = None
        self._current_green_time = 0
        self._episode = 0

    # ------------------- public API -------------------
    def run(self, episode: int, epsilon: float):
        sim_t0 = timeit.default_timer()
        self._episode = episode
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("D3QN Simulating...")

        # 初始化统计量
        self._step = 0
        self._waiting_times = {}       # vid -> accumulated waiting
        self._sum_neg_reward = 0.0
        self._sum_queue_length = 0.0
        self._sum_waiting_time = 0.0
        episode_reward = 0.0

        # 初始状态
        state = self._get_state()
        total_wait = self._collect_waiting_times()

        if episode == 0:
            print(f"[State Normalization] 模式: {self._state_normalization}")
            print(f"[State] 维度: {state.shape}, 值域: [{np.min(state):.3f}, {np.max(state):.3f}]")

        # 初次动作
        action = self._choose_action(state, epsilon, action_mask=None)
        self._set_green_phase(action)
        self._simulate(self._green)
        self._current_action = action
        self._current_green_time = self._green

        # 主循环
        while self._step < self._max_steps:
            next_state = self._get_state()
            next_total_wait = self._collect_waiting_times()

            # 基础奖励：等待时间变化（负向）
            waiting_time_change = total_wait - next_total_wait
            base_reward = waiting_time_change * self._waiting_time_penalty_scale

            # 拥堵惩罚
            congestion_penalty = self._check_congestion_penalty()

            # ★ 放行量正奖励：进口道上“在动”的车（近似通行）
            flow = self._throughput_proxy()
            flow_bonus = self._flow_w * float(flow)

            # 总奖励
            reward = base_reward + congestion_penalty + flow_bonus

            # Debug（前 100 步每 20 步打印一次）
            if self._step < 100 and self._step % 20 == 0:
                print(f"[Debug] step={self._step} Δwait={waiting_time_change:.2f} "
                      f"base={base_reward:.2f} cong={congestion_penalty:.2f} "
                      f"flow={flow} bonus={flow_bonus:.2f} total={reward:.2f}")

            episode_reward += reward
            done = int(self._step >= self._max_steps)

            # 存储样本
            self._Memory.add_sample((state, action, reward, next_state, done))
            if reward < 0:
                self._sum_neg_reward += reward
            if done:
                break

            # 基于工程约束的动作掩码
            action_mask = self._build_action_mask(action)
            next_action = self._choose_action(next_state, epsilon, action_mask=action_mask)

            # 切换相位 → 黄灯
            if next_action != action:
                self._set_yellow_phase(action)
                self._simulate(self._yellow)
                self._current_action = next_action
                self._current_green_time = 0

            # 执行下一相位绿灯
            self._set_green_phase(next_action)
            self._simulate(self._green)

            # 统计连续绿灯时长
            if self._current_action == next_action:
                self._current_green_time += self._green
            else:
                self._current_action = next_action
                self._current_green_time = self._green

            # 滚动
            state, action, total_wait = next_state, next_action, next_total_wait

        # 结束仿真
        self._save_episode_stats()
        self._episode_rewards.append(episode_reward)
        traci.close()
        simulation_time = round(timeit.default_timer() - sim_t0, 1)
        print(f"D3QN Total reward: {self._sum_neg_reward:.2f} - Epsilon: {epsilon:.3f}")

        # 训练
        train_t0 = timeit.default_timer()
        episode_loss = 0.0
        can_train = self._can_train_now()
        if can_train and self._epochs > 0:
            for _ in range(self._epochs):
                loss = self._replay_d3qn()
                if loss is not None:
                    episode_loss += loss
            self._episode_losses.append(episode_loss / max(1, self._epochs))
        else:
            self._episode_losses.append(0.0)

        training_time = round(timeit.default_timer() - train_t0, 1)
        return simulation_time, training_time

    def get_episode_steps(self):
        return self._step

    # ------------------- SUMO 交互 -------------------
    def _simulate(self, steps_todo: int):
        steps_todo = min(steps_todo, self._max_steps - self._step)
        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1

            qlen = self._get_queue_length()
            self._sum_queue_length += qlen
            # 用 halting 近似累计等待
            self._sum_waiting_time += qlen

    def _collect_waiting_times(self):
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for vid in car_list:
            wt = traci.vehicle.getAccumulatedWaitingTime(vid)
            road = traci.vehicle.getRoadID(vid)
            if road in incoming_roads:
                self._waiting_times[vid] = wt
            else:
                self._waiting_times.pop(vid, None)
        return float(sum(self._waiting_times.values()))

    def _choose_action(self, state, epsilon: float, action_mask=None):
        # 固定 ε（不在模型内部二次衰减），传入动作掩码
        return self._Model.select_action(
            state,
            steps_done=getattr(self._Memory, "steps_done", 0),
            eps_start=epsilon, eps_end=epsilon, eps_decay=1,
            action_mask=action_mask
        )

    def _set_yellow_phase(self, old_action: int):
        if old_action in (0, 1, 2, 3):
            traci.trafficlight.setPhase("TL", old_action * 2 + 1)

    def _set_green_phase(self, action_number: int):
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self) -> int:
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        return int(halt_N + halt_S + halt_E + halt_W)

    def _get_direction_queue_lengths(self) -> dict:
        return {
            'N': int(traci.edge.getLastStepHaltingNumber("N2TL")),
            'S': int(traci.edge.getLastStepHaltingNumber("S2TL")),
            'E': int(traci.edge.getLastStepHaltingNumber("E2TL")),
            'W': int(traci.edge.getLastStepHaltingNumber("W2TL"))
        }

    def _check_congestion_penalty(self) -> float:
        """检查拥堵并返回惩罚值（越界按比例）"""
        queue_lengths = self._get_direction_queue_lengths()
        max_queue = max(queue_lengths.values())
        if max_queue > self._congestion_threshold:
            penalty = self._congestion_penalty * (max_queue / max(1, self._congestion_threshold))
            self._congestion_penalties.append(penalty)
            return penalty
        return 0.0

    # ===== 放行量代理（新增） =====
    def _throughput_proxy(self) -> int:
        """
        以‘进口道上正在移动的车辆数’近似放行量（越多越好）
        moving = vehicle_number - halting_number
        """
        incoming = ["E2TL", "N2TL", "W2TL", "S2TL"]
        moving = 0
        for e in incoming:
            n = traci.edge.getLastStepVehicleNumber(e)
            h = traci.edge.getLastStepHaltingNumber(e)
            moving += max(0, int(n - h))
        return moving

    # ------------------- 状态编码 -------------------
    def _get_state(self) -> np.ndarray:
        """
        8 组 (W主/左, N主/左, E主/左, S主/左) × 10 cell -> 80 维
        支持：binary, count, count_normalized, z_score
        """
        state = np.zeros(self._num_states, dtype=np.float32)
        car_list = traci.vehicle.getIDList()

        for vid in car_list:
            lane_pos = traci.vehicle.getLanePosition(vid)
            lane_id = traci.vehicle.getLaneID(vid)

            if   lane_pos < 7:   cell = 0
            elif lane_pos < 14:  cell = 1
            elif lane_pos < 21:  cell = 2
            elif lane_pos < 28:  cell = 3
            elif lane_pos < 40:  cell = 4
            elif lane_pos < 60:  cell = 5
            elif lane_pos < 100: cell = 6
            elif lane_pos < 160: cell = 7
            elif lane_pos < 400: cell = 8
            else:                cell = 9

            if lane_id in ("W2TL_0", "W2TL_1", "W2TL_2"): group = 0
            elif lane_id == "W2TL_3":                     group = 1
            elif lane_id in ("N2TL_0", "N2TL_1", "N2TL_2"): group = 2
            elif lane_id == "N2TL_3":                     group = 3
            elif lane_id in ("E2TL_0", "E2TL_1", "E2TL_2"): group = 4
            elif lane_id == "E2TL_3":                     group = 5
            elif lane_id in ("S2TL_0", "S2TL_1", "S2TL_2"): group = 6
            elif lane_id == "S2TL_3":                     group = 7
            else:
                continue

            idx = group * 10 + cell
            if 0 <= idx < self._num_states:
                if self._state_normalization == "binary":
                    state[idx] = 1.0
                else:
                    state[idx] += 1.0

        # 归一化
        if self._state_normalization == "count_normalized":
            mean, std = np.mean(state), np.std(state)
            state = (state - mean) / std if std > 1e-8 else (state - mean)
            mn, mx = np.min(state), np.max(state)
            if mx - mn > 1e-8:
                state = (state - mn) / (mx - mn)
        elif self._state_normalization == "z_score":
            mean, std = np.mean(state), np.std(state)
            state = (state - mean) / std if std > 1e-8 else (state - mean)

        return state

    # ------------------- 动作掩码（约束） -------------------
    def _action_orientation(self, a: int) -> str:
        return "NS" if a in (0, 1) else "EW"

    def _served_actions_for_orientation(self, ori: str):
        return (0, 1) if ori == "NS" else (2, 3)

    def _ns_ew_queues(self):
        q = self._get_direction_queue_lengths()
        return q['N'] + q['S'], q['E'] + q['W']

    def _max_wait_by_orientation(self):
        ns_max = 0.0
        ew_max = 0.0
        for vid, wt in self._waiting_times.items():
            try:
                road = traci.vehicle.getRoadID(vid)
            except Exception:
                continue
            if road in ("N2TL", "S2TL"):
                ns_max = max(ns_max, wt)
            elif road in ("E2TL", "W2TL"):
                ew_max = max(ew_max, wt)
        return ns_max, ew_max

    def _build_action_mask(self, current_action: int | None):
        mask = np.ones(self._num_actions, dtype=bool)

        # 1) 最长绿灯：禁止继续同相位
        if current_action is not None and self._current_green_time >= self._max_green:
            mask[current_action] = False

        # 2) NS/EW 失衡：禁止继续服务错误方向
        ns, ew = self._ns_ew_queues()
        need_ns = (ns > self._imbalance_ratio * max(1, ew))
        need_ew = (ew > self._imbalance_ratio * max(1, ns))
        if current_action is not None:
            if need_ns and self._action_orientation(current_action) == "EW":
                mask[current_action] = False
            if need_ew and self._action_orientation(current_action) == "NS":
                mask[current_action] = False

        # 3) 最大累计等待：必须开放对应方向
        ns_max, ew_max = self._max_wait_by_orientation()
        if ns_max > self._max_wait_threshold:
            mask[list(self._served_actions_for_orientation("NS"))] = True
            if current_action is not None and self._action_orientation(current_action) == "EW":
                mask[current_action] = False
        if ew_max > self._max_wait_threshold:
            mask[list(self._served_actions_for_orientation("EW"))] = True
            if current_action is not None and self._action_orientation(current_action) == "NS":
                mask[current_action] = False

        if not mask.any():
            mask[:] = True
        return mask

    # ------------------- replay (Double + PER) -------------------
    def _replay_d3qn(self):
        batch, indices, weights = self._Memory.get_samples(self._Model.batch_size)
        if len(batch) == 0:
            return None

        states      = np.asarray([b[0] for b in batch], dtype=np.float32)
        actions     = np.asarray([b[1] for b in batch], dtype=np.int32)
        rewards     = np.asarray([b[2] for b in batch], dtype=np.float32)
        next_states = np.asarray([b[3] for b in batch], dtype=np.float32)
        dones       = np.asarray([b[4] if len(b) >= 5 else 0.0 for b in batch], dtype=np.float32)

        y, td_errors = self._Model.compute_double_dqn_targets(
            states, actions, rewards, next_states, dones, action_mask_next=None, gamma=self._gamma
        )

        loss = self._Model.train_batch(states, y, weights=weights)

        if indices is not None and hasattr(self._Memory, "update_priorities"):
            self._Memory.update_priorities(indices, np.abs(td_errors) + 1e-6)

        if getattr(self._Model, "update_gamma", False):
            self._Model.learn_gamma()

        return loss

    # ------------------- stats -------------------
    def _save_episode_stats(self):
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / max(1, self._max_steps))
        self._congestion_penalties = []

    # getters
    @property
    def reward_store(self): return self._reward_store
    @property
    def cumulative_wait_store(self): return self._cumulative_wait_store
    @property
    def avg_queue_length_store(self): return self._avg_queue_length_store
    @property
    def episode_rewards(self): return self._episode_rewards
    @property
    def episode_losses(self): return self._episode_losses

    # ------------------- helpers -------------------
    def _can_train_now(self) -> bool:
        if self._warmup_min is not None:
            return len(self._Memory) >= self._warmup_min
        size_min = getattr(self._Memory, "size_min", None)
        if size_min is not None:
            return len(self._Memory) >= int(size_min)
        return True
