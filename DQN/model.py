import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys
import random
import math

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class ImprovedTrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, 
                 use_sgd=False, target_update_freq=3000):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._use_sgd = use_sgd
        self._target_update_freq = target_update_freq
        self._learn_steps = 0
        
        # 新增：损失记录
        self._loss_history = []
        self._episode_losses = []
        
        # 创建策略网络和目标网络
        self._policy_model = self._build_model(num_layers, width)
        self._target_model = self._build_model(num_layers, width)
        self._target_model.set_weights(self._policy_model.get_weights())
        
        # 用于动态γ调整
        self._gamma = 0.75
        self._fixed_gamma = 0.75
        self._update_gamma = False
        self._z = None

    def _build_model(self, num_layers, width):
        """
        构建改进的深度神经网络
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='leaky_relu', 
                        kernel_initializer='glorot_uniform')(inputs)
        
        for _ in range(num_layers - 1):
            x = layers.Dense(width, activation='leaky_relu',
                           kernel_initializer='glorot_uniform')(x)
        
        outputs = layers.Dense(self._output_dim, activation='linear',
                              kernel_initializer='glorot_uniform')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='improved_model')
        
        # 选择优化器
        if self._use_sgd:
            optimizer = SGD(learning_rate=self._learning_rate)
        else:
            optimizer = Adam(learning_rate=self._learning_rate)
            
        model.compile(loss=losses.MeanSquaredError(), optimizer=optimizer)
        return model

    def predict_one(self, state):
        """
        预测单个状态的Q值
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._policy_model.predict(state, verbose=0)

    def predict_batch(self, states):
        """
        预测批量状态的Q值
        """
        return self._policy_model.predict(states, verbose=0)

    def predict_target_batch(self, states):
        """
        使用目标网络预测Q值
        """
        return self._target_model.predict(states, verbose=0)

    def train_batch(self, states, q_sa):
        """
        训练网络
        """
        # 梯度裁剪
        with tf.GradientTape() as tape:
            predictions = self._policy_model(states)
            loss = tf.keras.losses.MeanSquaredError()(q_sa, predictions)
        
        # 记录损失
        loss_value = float(loss.numpy())
        self._loss_history.append(loss_value)
        self._episode_losses.append(loss_value)
        
        gradients = tape.gradient(loss, self._policy_model.trainable_variables)
        # 梯度裁剪到[-1, 1]
        gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
        
        self._policy_model.optimizer.apply_gradients(
            zip(gradients, self._policy_model.trainable_variables)
        )
        
        self._learn_steps += 1
        
        # 定期更新目标网络
        if self._learn_steps % self._target_update_freq == 0:
            self._target_model.set_weights(self._policy_model.get_weights())

    def start_new_episode(self):
        """
        开始新的episode，清空当前episode的损失记录
        """
        if self._episode_losses:
            # 计算当前episode的平均损失
            avg_episode_loss = np.mean(self._episode_losses)
            # 可以选择保存episode平均损失到历史中
            # self._episode_avg_losses.append(avg_episode_loss)
        self._episode_losses = []

    def get_loss_history(self):
        """
        获取完整的损失历史
        """
        return self._loss_history.copy()

    def get_episode_losses(self):
        """
        获取当前episode的损失
        """
        return self._episode_losses.copy()

    def get_recent_losses(self, n=100):
        """
        获取最近n次的损失值
        """
        return self._loss_history[-n:] if len(self._loss_history) >= n else self._loss_history.copy()

    def select_action(self, state, steps_done, invalid_action=None, eps_start=1.0, eps_end=0.1, eps_decay=83000):
        """
        改进的动作选择策略，包含拥堵感知和无效动作处理
        """
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
        
        if sample > eps_threshold:
            # 利用：选择最优动作
            q_values = self.predict_one(state)[0]
            sorted_indices = np.argsort(q_values)[::-1]  # 降序排列
            
            if invalid_action is not None and invalid_action:
                # 如果当前动作无效，选择第二优动作
                return sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
            else:
                return sorted_indices[0]
        else:
            # 探索：智能随机选择
            if invalid_action is not None and not invalid_action:
                # 拥堵感知：优先选择拥堵车道的动作
                congest_actions = self._get_congest_actions(state)
                if len(congest_actions) > 0:
                    return random.choice(congest_actions)
            
            return random.randint(0, self._output_dim - 1)

    def _get_congest_actions(self, state):
        """
        识别拥堵车道对应的动作
        """
        # 这里需要根据具体的状态编码来识别拥堵车道
        # 假设状态向量中高值表示拥堵
        congest_threshold = 0.8
        congest_lanes = [i for i, s in enumerate(state) if s > congest_threshold]
        
        # 映射车道到动作（需要根据具体环境调整）
        lane_to_action = {
            0: 0,  # 北向直行
            1: 1,  # 北向左转
            2: 2,  # 东向直行
            3: 3,  # 东向左转
            # 可以根据实际环境添加更多映射
        }
        
        return [lane_to_action[lane] for lane in congest_lanes if lane in lane_to_action]

    def learn_gamma(self):
        """
        动态调整γ值
        """
        # 这里可以实现动态γ调整逻辑
        # 基于当前训练状态调整γ值
        if self._learn_steps > 10000:
            # 随着训练进行，逐渐增加γ值
            self._gamma = min(0.95, self._fixed_gamma + 0.001 * (self._learn_steps - 10000) / 1000)
        self._update_gamma = False

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        self._policy_model.save(os.path.join(path, 'improved_trained_model.keras'))
        plot_model(self._policy_model, to_file=os.path.join(path, 'improved_model_structure.png'),
                show_shapes=True, show_layer_names=True)
        
        # 保存训练参数和损失历史
        np.save(os.path.join(path, 'improved_training_params.npy'), {
            'learn_steps': self._learn_steps,
            'gamma': self._gamma,
            'fixed_gamma': self._fixed_gamma,
            'loss_history': self._loss_history
        })
        
        # 单独保存损失历史为文本文件，便于查看
        loss_file = os.path.join(path, 'loss_history.txt')
        with open(loss_file, 'w') as f:
            for i, loss in enumerate(self._loss_history):
                f.write(f"{i+1}\t{loss}\n")
        print(f"损失历史已保存到: {loss_file}")

    def load_model(self, path):
        keras_path = os.path.join(path, 'improved_trained_model.keras')
        saved_dir  = os.path.join(path, 'improved_trained_model_savedmodel')
        h5_path    = os.path.join(path, 'improved_trained_model.h5')

        custom_objects = {}  # 如果你把 Lambda 改成自定义层/具名函数，放到这里

        if os.path.exists(keras_path):
            self._policy_model = tf.keras.models.load_model(keras_path, compile=False, custom_objects=custom_objects)
        elif os.path.isdir(saved_dir):
            self._policy_model = tf.keras.models.load_model(saved_dir, compile=False, custom_objects=custom_objects)
        elif os.path.exists(h5_path):
            # 最后兜底：旧h5（可能会失败）
            self._policy_model = tf.keras.models.load_model(h5_path, compile=False, custom_objects=custom_objects)
        else:
            raise FileNotFoundError(f"No model found in {path}")

        self._target_model.set_weights(self._policy_model.get_weights())

        params_path = os.path.join(path, 'improved_training_params.npy')
        if os.path.exists(params_path):
            params = np.load(params_path, allow_pickle=True).item()
            self._learn_steps = params.get('learn_steps', 0)
            self._gamma = params.get('gamma', 0.99)
            self._fixed_gamma = params.get('fixed_gamma', 0.99)
            # 加载损失历史
            self._loss_history = params.get('loss_history', [])
            print(f"已加载损失历史，共 {len(self._loss_history)} 条记录")
        self._policy_model.compile(loss=losses.MeanSquaredError(), optimizer=Adam(learning_rate=self._learning_rate))
        self._target_model.compile(loss=losses.MeanSquaredError(), optimizer=Adam(learning_rate=self._learning_rate))


    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def gamma(self):
        return self._gamma

    @property
    def learn_steps(self):
        return self._learn_steps

    @property
    def update_gamma(self):
        return self._update_gamma

    @property
    def loss_history(self):
        return self._loss_history

    @property
    def episode_losses(self):
        return self._episode_losses


class ImprovedMemory:
    def __init__(self, size_max, size_min):
        self._samples = []
        self._size_max = size_max
        self._size_min = size_min
        self._steps_done = 0

    def add_sample(self, sample):
        """
        添加经验样本
        """
        self._samples.append(sample)
        self._steps_done += 1
        
        if self._size_now() > self._size_max:
            self._samples.pop(0)  # 移除最旧的样本

    def get_samples(self, n):
        """
        获取随机样本
        """
        if self._size_now() < self._size_min:
            return []

        if n > self._size_now():
            return random.sample(self._samples, self._size_now())
        else:
            return random.sample(self._samples, n)

    def _size_now(self):
        """
        当前内存大小
        """
        return len(self._samples)

    @property
    def steps_done(self):
        return self._steps_done 

class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'improved_trained_model.keras')
        
        print(f"Looking for model file at: {model_file_path}")
        print(f"Model file exists: {os.path.isfile(model_file_path)}")
        print(f"Model folder path: {model_folder_path}")
        
        if os.path.isfile(model_file_path):
            print("Loading model...")
            loaded_model = load_model(model_file_path)
            print("Model loaded successfully!")
            return loaded_model
        else:
            # 尝试不同的路径组合
            alt_paths = [
                os.path.join(model_folder_path, 'improved_trained_model.keras'),
                os.path.join(model_folder_path, 'trained_model.keras'),
                os.path.join(model_folder_path, 'model.keras'),
                os.path.join(model_folder_path, 'model.h5'),
                os.path.join(model_folder_path, 'trained_model.h5')
            ]
            
            for alt_path in alt_paths:
                if os.path.isfile(alt_path):
                    print(f"Found model at alternative path: {alt_path}")
                    loaded_model = load_model(alt_path)
                    print("Model loaded successfully from alternative path!")
                    return loaded_model
            
            print(f"Available files in {model_folder_path}:")
            if os.path.isdir(model_folder_path):
                for item in os.listdir(model_folder_path):
                    print(f"  - {item}")
            
            sys.exit(f"Model file not found. Checked paths: {model_file_path}")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim