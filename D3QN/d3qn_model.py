import os 
import math
import random
import numpy as np
import tensorflow as tf
import sys

from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


# --------- Dueling: mean advantage layer ---------
@tf.keras.utils.register_keras_serializable()
class MeanAdvantage(layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1, keepdims=True)


# --------- D3QN Model (Dueling + Double) ---------
class D3QNModel:
    def __init__(
        self,
        num_layers: int,
        width: int,
        batch_size: int,
        learning_rate: float,
        input_dim: int,
        output_dim: int,
        target_update_freq: int = 1000,
        dueling: bool = True,
        target_value_clip_min: float = -50.0,
        target_value_clip_max: float = 50.0,
    ):
        self._input_dim = int(input_dim)
        self._output_dim = int(output_dim)
        self._batch_size = int(batch_size)
        self._lr = float(learning_rate)
        self._target_update_freq = int(target_update_freq)
        self._dueling = bool(dueling)
        self._target_value_clip_min = float(target_value_clip_min)
        self._target_value_clip_max = float(target_value_clip_max)

        self._learn_steps = 0
        self._gamma = 0.99         # 默认训练期 γ
        self._fixed_gamma = 0.99
        self._update_gamma = False  # 外部可置 True 触发 learn_gamma()

        self._policy_model = self._build_model(num_layers, width)
        self._target_model = self._build_model(num_layers, width)
        self._target_model.set_weights(self._policy_model.get_weights())

    # --------- network ---------
    def _build_model(self, num_layers: int, width: int) -> keras.Model:
        inp = keras.Input(shape=(self._input_dim,), name="obs")
        x = layers.Dense(width, activation="relu", kernel_initializer="glorot_uniform")(inp)
        for _ in range(num_layers - 1):
            x = layers.Dense(width, activation="relu", kernel_initializer="glorot_uniform")(x)

        if self._dueling:
            v = layers.Dense(width // 2, activation="relu")(x)
            v = layers.Dense(1, activation=None, name="value")(v)
            a = layers.Dense(width // 2, activation="relu")(x)
            a = layers.Dense(self._output_dim, activation=None, name="advantage")(a)
            a_mean = MeanAdvantage()(a)
            a_centered = layers.Subtract()([a, a_mean])
            out = layers.Add()([v, a_centered])
        else:
            out = layers.Dense(self._output_dim, activation="linear", kernel_initializer="glorot_uniform")(x)

        model = keras.Model(inputs=inp, outputs=out, name="d3qn")
        # 梯度裁剪更稳
        opt = Adam(learning_rate=self._lr, clipnorm=10.0)
        model.compile(loss=losses.Huber(), optimizer=opt)
        return model

    # --------- predicts ---------
    def predict_one(self, state: np.ndarray) -> np.ndarray:
        state = np.reshape(state, [1, self._input_dim])
        return self._policy_model.predict(state, verbose=0)

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        return self._policy_model.predict(states, verbose=0)

    def predict_target_batch(self, states: np.ndarray) -> np.ndarray:
        return self._target_model.predict(states, verbose=0)

    # --------- Double DQN targets & train ---------
    def compute_double_dqn_targets(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        action_mask_next: np.ndarray | None = None,
        gamma: float | None = None,
    ):
        """Return (Y targets, TD-errors) with Double DQN."""
        g = float(self._gamma if gamma is None else gamma)

        q_s = self.predict_batch(states)                     # [B, A]
        q_next_policy = self.predict_batch(next_states)      # [B, A]
        if action_mask_next is not None:
            # 屏蔽不可用动作
            q_next_policy = q_next_policy - (1 - action_mask_next) * 1e9
        a_star = np.argmax(q_next_policy, axis=1)            # [B]

        q_next_target = self.predict_target_batch(next_states)  # [B, A]
        bootstrap = q_next_target[np.arange(len(a_star)), a_star]  # [B]

        targets_sa = rewards + (1.0 - dones) * g * bootstrap
        
        # 数值剪裁：限制目标值范围，防止过大或过小
        targets_sa = np.clip(targets_sa, self._target_value_clip_min, self._target_value_clip_max)
        
        y = q_s.copy()
        y[np.arange(len(actions)), actions] = targets_sa

        td_errors = targets_sa - q_s[np.arange(len(actions)), actions]
        return y, td_errors

    def train_batch(self, states: np.ndarray, q_targets: np.ndarray, weights: np.ndarray | None = None) -> float:
        """Train once; return keras loss (float)."""
        history = self._policy_model.fit(
            states,
            q_targets,
            batch_size=self._batch_size,
            epochs=1,
            verbose=0,
            sample_weight=weights,
        )
        self._learn_steps += 1
        if self._learn_steps % self._target_update_freq == 0:
            self._target_model.set_weights(self._policy_model.get_weights())
        return float(history.history["loss"][0])

    # --------- action selection (supports action mask) ---------
    def select_action(
        self,
        state: np.ndarray,
        steps_done: int,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: int = 10000,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """
        - 若 eps_start==eps_end 或 eps_decay<=1，则视作“固定 ε”。
        - action_mask: 0/1，有效动作为1。
        """
        if eps_decay <= 1 or abs(eps_start - eps_end) < 1e-12:
            eps = float(eps_start)
        else:
            eps = eps_end + (eps_start - eps_end) * math.exp(-steps_done / float(eps_decay))

        if random.random() < eps:  # explore
            if action_mask is not None and np.any(action_mask):
                valid = np.where(action_mask == 1)[0]
                return int(np.random.choice(valid)) if len(valid) > 0 else int(np.random.randint(self._output_dim))
            return int(np.random.randint(self._output_dim))

        q = self.predict_one(state)[0]
        if action_mask is not None:
            q = q.copy()
            q[action_mask == 0] = -1e9
        return int(np.argmax(q))

    # --------- optional gamma schedule ---------
    def learn_gamma(self):
        if self._learn_steps > 5000:
            self._gamma = min(0.99, self._fixed_gamma + 0.0001 * (self._learn_steps - 5000) / 1000.0)
        self._update_gamma = False

    # --------- save / load ---------
    def save_model(self, path: str):
        os.makedirs(path, exist_ok=True)
        self._policy_model.save(os.path.join(path, "d3qn_trained_model.keras"))
        try:
            plot_model(
                self._policy_model,
                to_file=os.path.join(path, "d3qn_model_structure.png"),
                show_shapes=True,
                show_layer_names=True,
            )
        except Exception:
            pass
        np.save(
            os.path.join(path, "d3qn_training_params.npy"),
            {
                "learn_steps": self._learn_steps,
                "gamma": self._gamma,
                "fixed_gamma": self._fixed_gamma,
                "dueling": self._dueling,
                "target_update_freq": self._target_update_freq,
                "input_dim": self._input_dim,
                "output_dim": self._output_dim,
            },
            allow_pickle=True,
        )

    def load_model(self, path: str):
        custom_objects = {"MeanAdvantage": MeanAdvantage}
        tried = []
        for name in ("d3qn_trained_model.keras", "d3qn_trained_model_savedmodel", "d3qn_trained_model.h5"):
            f = os.path.join(path, name)
            tried.append(f)
            if os.path.exists(f) or os.path.isdir(f):
                try:
                    self._policy_model = tf.keras.models.load_model(f, custom_objects=custom_objects, compile=False)
                    break
                except Exception:
                    continue
        else:
            raise FileNotFoundError(f"No model found in: {path} (tried: {tried})")

        self._target_model.set_weights(self._policy_model.get_weights())
        opt = Adam(learning_rate=self._lr, clipnorm=10.0)
        self._policy_model.compile(loss=losses.Huber(), optimizer=opt)
        self._target_model.compile(loss=losses.Huber(), optimizer=opt)

        params_path = os.path.join(path, "d3qn_training_params.npy")
        if os.path.exists(params_path):
            params = np.load(params_path, allow_pickle=True).item()
            self._learn_steps = int(params.get("learn_steps", 0))
            self._gamma = float(params.get("gamma", 0.99))
            self._fixed_gamma = float(params.get("fixed_gamma", 0.99))
            self._dueling = bool(params.get("dueling", True))

    # --------- properties ---------
    @property
    def input_dim(self): return self._input_dim
    @property
    def output_dim(self): return self._output_dim
    @property
    def batch_size(self): return self._batch_size
    @property
    def gamma(self): return self._gamma
    @property
    def learn_steps(self): return self._learn_steps
    @property
    def update_gamma(self): return self._update_gamma


# --------- Prioritized Replay Memory (PER) ---------
class PrioritizedReplayMemory:
    def __init__(
        self,
        size_max: int,
        size_min: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 4e-3,      # ★ 更快退火（原 1e-3 -> 4e-3）
        epsilon_priority: float = 1e-6,
        uniform_mix_ratio: float = 0.2,    # ★ 新增：混合采样比例（20% 均匀）
        priority_clip_max: float | None = None,  # 可选：裁剪极端优先级
    ):
        self._samples: list[tuple] = []
        self._priorities: list[float] = []
        self._size_max = int(size_max)
        self._size_min = int(size_min)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._beta_inc = float(beta_increment)
        self._eps = float(epsilon_priority)
        self._steps_done = 0
        self._max_priority = 1.0
        self._mix = float(np.clip(uniform_mix_ratio, 0.0, 0.9))
        self._p_clip = None if priority_clip_max is None else float(priority_clip_max)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def size_min(self) -> int:
        return self._size_min

    @property
    def steps_done(self) -> int:
        return self._steps_done

    def add_sample(self, sample: tuple, priority: float | None = None):
        p = self._max_priority if priority is None else max(float(priority), self._eps)
        if self._p_clip is not None:
            p = min(p, self._p_clip)
        self._samples.append(sample)
        self._priorities.append(p)
        self._steps_done += 1
        if len(self._samples) > self._size_max:
            self._samples.pop(0)
            self._priorities.pop(0)

    def get_samples(self, n: int):
        if len(self) < self._size_min:
            return [], [], None

        n = int(min(max(1, n), len(self)))
        pr = np.asarray(self._priorities, dtype=np.float32)
        pr = np.maximum(pr, self._eps)

        probs = pr ** self._alpha
        s = probs.sum()
        if not np.isfinite(s) or s <= 0:
            probs = np.ones_like(probs) / len(pr)
        else:
            probs = probs / s

        # ★ 混合采样：部分按 PER，部分均匀
        n_uni = int(round(self._mix * n))
        n_per = n - n_uni

        # PER 部分（不放回）
        idx_per = np.random.choice(len(self), size=n_per, p=probs, replace=False)

        # 均匀部分（不放回且避免重复）
        remaining = np.setdiff1d(np.arange(len(self)), idx_per, assume_unique=False)
        if n_uni > 0 and len(remaining) > 0:
            n_uni = min(n_uni, len(remaining))
            idx_uni = np.random.choice(remaining, size=n_uni, replace=False)
            idx = np.concatenate([idx_per, idx_uni])
        else:
            idx = idx_per

        # 重要性采样权重（仍用 PER 概率，以保持修正）
        w = (len(self) * probs[idx]) ** (-self._beta)
        w /= w.max() if w.size > 0 else 1.0

        # ★ 更快退火
        self._beta = float(min(1.0, self._beta + self._beta_inc))

        batch = [self._samples[i] for i in idx]
        return batch, idx, w.astype(np.float32)

    def update_priorities(self, indices, priorities):
        idx = np.asarray(indices, dtype=int)
        pr = np.maximum(np.abs(np.asarray(priorities, dtype=float)), self._eps)  # ★ abs(TD)+eps
        if self._p_clip is not None:
            pr = np.minimum(pr, self._p_clip)
        for i, p in zip(idx, pr):
            if 0 <= i < len(self._priorities):
                self._priorities[i] = float(p)
                if p > self._max_priority:
                    self._max_priority = float(p)

class TestModel:
    def __init__(self, input_dim: int, model_path: str):
        self._input_dim = int(input_dim)
        self._model = self._load_model(model_path)

    def _load_model(self, model_folder_path: str):
        custom_objects = {"MeanAdvantage": MeanAdvantage}
        tried = []

        main = os.path.join(model_folder_path, "d3qn_trained_model.keras")
        tried.append(main)

        alt_files = [
            "improved_trained_model.keras",
            "trained_model.keras",
            "model.keras",
            "model.h5",
            "trained_model.h5",
        ]
        for name in alt_files:
            tried.append(os.path.join(model_folder_path, name))

        savedmodel_dir = os.path.join(model_folder_path, "d3qn_trained_model_savedmodel")
        tried.append(savedmodel_dir)

        for p in tried:
            if os.path.isfile(p) or os.path.isdir(p):
                print(f"[TestModel] Loading model from: {p}")
                m = load_model(p, custom_objects=custom_objects, compile=False)
                print("[TestModel] Loaded.")
                return m

        print(f"[TestModel] Available in {model_folder_path}:")
        if os.path.isdir(model_folder_path):
            for it in os.listdir(model_folder_path):
                print(" -", it)
        sys.exit(f"[TestModel] Model not found. Tried: {tried}")

    def predict_one(self, state: np.ndarray):
        state = np.reshape(state, (1, self._input_dim)).astype(np.float32, copy=False)
        return self._model.predict(state, verbose=0)

    def predict_batch(self, states: np.ndarray):
        states = np.asarray(states, dtype=np.float32)
        return self._model.predict(states, verbose=0)

    @property
    def input_dim(self): return self._input_dim