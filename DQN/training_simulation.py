import traci 
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class ImprovedSimulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        
        # 新增：无效动作处理
        self._invalid_action = False
        self._last_action = -1
        
        # 新增：损失记录
        self._episode_losses = []
        self._avg_loss_store = []

    def run(self, episode, epsilon):
        """
        运行一个episode的仿真，然后开始训练
        """
        start_time = timeit.default_timer()

        # 新增：开始新的episode，初始化损失记录
        self._Model.start_new_episode()
        self._episode_losses = []

        # 生成路线文件并设置SUMO
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # 初始化
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1
        self._invalid_action = False

        while self._step < self._max_steps:
            # 获取当前交叉口状态
            current_state = self._get_state()

            # 计算前一个动作的奖励
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # 将数据保存到记忆中
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # 选择动作（使用改进的动作选择策略）
            action = self._choose_action(current_state, epsilon)

            # 如果选择的相位与上一个不同，激活黄灯相位
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # 执行选择的相位
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # 保存变量供后续使用
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # 只保存有意义的负奖励
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        
        # 改进的训练循环
        for _ in range(self._training_epochs):
            self._replay()
            # 动态调整γ值
            if self._Model.update_gamma:
                self._Model.learn_gamma()
        
        # 新增：收集当前episode的损失统计
        current_episode_losses = self._Model.get_episode_losses()
        if current_episode_losses:
            avg_episode_loss = np.mean(current_episode_losses)
            self._episode_losses.append(avg_episode_loss)
            print(f"Episode {episode+1} 平均损失: {avg_episode_loss:.6f}")
        
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def _simulate(self, steps_todo):
        """
        执行仿真步骤
        """
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length  # 1 step while waiting in queue means 1 second waited, for each car, therefore queue_length == waited_seconds

    def _collect_waiting_times(self):
        """
        收集所有进入车道的等待时间
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road = traci.vehicle.getRoadID(car_id)
            if road in incoming_roads:
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state, epsilon):
        """
        使用改进的动作选择策略
        """
        # 使用模型的智能动作选择
        return self._Model.select_action(
            state, 
            self._Memory.steps_done, 
            self._invalid_action,
            eps_start=1.0,
            eps_end=0.1,
            eps_decay=83000
        )

    def _set_yellow_phase(self, old_action):
        """
        在SUMO中激活正确的黄灯组合
        """
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        在SUMO中激活正确的绿灯组合
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        """
        获取每个进入车道中速度为0的车辆数量
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_state(self):
        """
        从SUMO获取交叉口状态，以单元格占用率的形式
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos

            # 距离交通灯的距离（米）-> 映射到单元格
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # 找到车辆所在的车道
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False

            if valid_car:
                state[car_position] = 1

        return state

    def _replay(self):
        """
        从记忆中检索一组样本，对每个样本更新学习方程，然后训练
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch])

            # 预测
            q_s_a = self._Model.predict_batch(states)
            q_s_a_d = self._Model.predict_target_batch(next_states)  # 使用目标网络

            # 设置训练数组
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]
                current_q = q_s_a[i]
                # 使用动态γ值
                current_q[action] = reward + self._Model.gamma * np.amax(q_s_a_d[i])
                x[i] = state
                y[i] = current_q

            self._Model.train_batch(x, y)

    def _save_episode_stats(self):
        """
        保存episode的统计信息
        """
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)
        
        # 新增：保存损失统计
        if self._episode_losses:
            self._avg_loss_store.append(self._episode_losses[-1])

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

    @property
    def avg_loss_store(self):
        return self._avg_loss_store

    @property
    def episode_losses(self):
        return self._episode_losses 