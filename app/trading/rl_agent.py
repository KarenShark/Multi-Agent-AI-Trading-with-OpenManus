"""
强化学习智能体

基于深度强化学习的交易智能体，包括：
- PPO (Proximal Policy Optimization) Agent
- SAC (Soft Actor-Critic) Agent
- Custom Trading Agent
- Multi-Agent System
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
import gymnasium as gym
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.trading.trading_env import TradingEnvironment, create_trading_environment

class PolicyNetwork(nn.Module):
    """策略网络"""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = "relu"
    ):
        super(PolicyNetwork, self).__init__()

        self.activation_fn = getattr(torch.nn.functional, activation)

        # 构建网络层
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 使用LayerNorm替代BatchNorm避免batch_size=1的问题
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for i, layer in enumerate(self.network):
            if isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Dropout)):
                x = layer(x)
                # 在Linear层后应用激活函数（除了最后一层）
                if isinstance(layer, nn.Linear) and i < len(self.network) - 1:
                    x = self.activation_fn(x)

        # 最后一层使用softmax获得动作概率
        return torch.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """价值网络"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = "relu"
    ):
        super(ValueNetwork, self).__init__()

        self.activation_fn = getattr(torch.nn.functional, activation)

        # 构建网络层
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 使用LayerNorm替代BatchNorm避免batch_size=1的问题
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        # 输出层 (单一价值)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for i, layer in enumerate(self.network):
            if isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Dropout)):
                x = layer(x)
                # 在Linear层后应用激活函数（除了最后一层）
                if isinstance(layer, nn.Linear) and i < len(self.network) - 1:
                    x = self.activation_fn(x)

        return x


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 智能体

    适用于连续状态空间和离散动作空间的交易环境
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_symbols: int = 1,
        lr_policy: float = 3e-4,
        lr_value: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_symbols = n_symbols
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)

        # 网络
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)

        # 优化器
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)

        # 经验缓冲区
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }

        # 训练统计
        self.training_stats = {
            'episodes': 0,
            'policy_losses': [],
            'value_losses': [],
            'total_rewards': [],
            'episode_lengths': []
        }

        print(f"🧠 PPO智能体初始化:")
        print(f"  📊 状态维度: {state_dim}")
        print(f"  🎮 动作维度: {action_dim}")
        print(f"  💼 股票数量: {n_symbols}")
        print(f"  🖥️ 设备: {device}")

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 获取动作概率
            action_probs = self.policy_net(state_tensor)

            # 获取状态价值
            value = self.value_net(state_tensor)

        if training:
            # 训练模式：从概率分布中采样
            if self.n_symbols == 1:
                # 单股票情况
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                action_array = np.array([action.item()])
            else:
                # 多股票情况：为每只股票独立采样
                actions = []
                log_probs = []

                # 假设动作概率是(batch_size, n_symbols * action_per_symbol)
                actions_per_symbol = self.action_dim // self.n_symbols

                for i in range(self.n_symbols):
                    start_idx = i * actions_per_symbol
                    end_idx = (i + 1) * actions_per_symbol
                    symbol_probs = action_probs[0, start_idx:end_idx]

                    dist = Categorical(symbol_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                    actions.append(action.item())
                    log_probs.append(log_prob)

                action_array = np.array(actions)
                log_prob = torch.stack(log_probs).sum()
        else:
            # 推理模式：选择最大概率动作
            if self.n_symbols == 1:
                action = torch.argmax(action_probs, dim=1)
                action_array = np.array([action.item()])
                log_prob = torch.log(torch.max(action_probs))
            else:
                actions = []
                actions_per_symbol = self.action_dim // self.n_symbols

                for i in range(self.n_symbols):
                    start_idx = i * actions_per_symbol
                    end_idx = (i + 1) * actions_per_symbol
                    symbol_probs = action_probs[0, start_idx:end_idx]
                    action = torch.argmax(symbol_probs)
                    actions.append(action.item())

                action_array = np.array(actions)
                log_prob = torch.log(torch.max(action_probs))

        return action_array, log_prob, value.squeeze()

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool
    ):
        """存储转移经验"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value)
        self.memory['dones'].append(done)

    def update(self) -> Dict[str, float]:
        """更新策略和价值网络"""
        if len(self.memory['states']) == 0:
            return {'policy_loss': 0, 'value_loss': 0}

        # 转换为张量
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = torch.LongTensor(np.array(self.memory['actions'])).to(self.device)
        rewards = torch.FloatTensor(self.memory['rewards']).to(self.device)
        old_log_probs = torch.stack(self.memory['log_probs']).to(self.device)
        old_values = torch.stack(self.memory['values']).to(self.device)
        dones = torch.BoolTensor(self.memory['dones']).to(self.device)

        # 计算优势函数
        advantages, returns = self._compute_advantages(rewards, old_values, dones)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO更新
        policy_losses = []
        value_losses = []

        for epoch in range(self.k_epochs):
            # 重新计算动作概率
            action_probs = self.policy_net(states)
            new_values = self.value_net(states).squeeze()

            # 计算新的log概率
            if self.n_symbols == 1:
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions.squeeze())
                entropy = dist.entropy().mean()
            else:
                new_log_probs_list = []
                entropy_list = []
                actions_per_symbol = self.action_dim // self.n_symbols

                for i in range(self.n_symbols):
                    start_idx = i * actions_per_symbol
                    end_idx = (i + 1) * actions_per_symbol
                    symbol_probs = action_probs[:, start_idx:end_idx]

                    dist = Categorical(symbol_probs)
                    symbol_actions = actions[:, i] if len(actions.shape) > 1 else actions
                    new_log_prob = dist.log_prob(symbol_actions)

                    new_log_probs_list.append(new_log_prob)
                    entropy_list.append(dist.entropy())

                new_log_probs = torch.stack(new_log_probs_list).sum(dim=0)
                entropy = torch.stack(entropy_list).mean()

            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            # 价值损失
            value_loss = nn.MSELoss()(new_values, returns)

            # 更新网络
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        # 清空经验缓冲区
        self._clear_memory()

        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)

        # 记录统计信息
        self.training_stats['policy_losses'].append(avg_policy_loss)
        self.training_stats['value_losses'].append(avg_value_loss)

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss
        }

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算优势函数 (GAE)"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        next_value = 0  # 假设最后状态的价值为0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step].float()
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[step].float()
                next_value = values[step + 1]

            delta = rewards[step] + self.gamma * next_value * next_non_terminal - values[step]
            gae = delta + self.gamma * 0.95 * next_non_terminal * gae  # lambda = 0.95
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]

        return advantages, returns

    def _clear_memory(self):
        """清空经验缓冲区"""
        for key in self.memory:
            self.memory[key].clear()

    def train_on_environment(
        self,
        env: TradingEnvironment,
        num_episodes: int = 100,
        max_steps_per_episode: int = 1000,
        update_frequency: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """在交易环境中训练智能体"""
        print(f"🎓 开始PPO训练 ({num_episodes}轮)...")

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_length = 0

            for step in range(max_steps_per_episode):
                # 选择动作
                action, log_prob, value = self.select_action(state, training=True)

                # 执行动作
                next_state, reward, terminated, truncated, info = env.step(action)

                # 存储经验
                done = terminated or truncated
                self.store_transition(state, action, reward, log_prob, value, done)

                # 更新状态
                state = next_state
                episode_reward += reward
                episode_length += 1

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # 更新网络
            if (episode + 1) % update_frequency == 0:
                losses = self.update()

                if verbose:
                    recent_rewards = episode_rewards[-update_frequency:]
                    avg_reward = np.mean(recent_rewards)
                    print(f"  Episode {episode + 1}: 平均奖励={avg_reward:.3f}, "
                          f"策略损失={losses['policy_loss']:.4f}, "
                          f"价值损失={losses['value_loss']:.4f}")

            # 记录统计信息
            self.training_stats['episodes'] = episode + 1
            self.training_stats['total_rewards'] = episode_rewards
            self.training_stats['episode_lengths'] = episode_lengths

        print(f"✅ PPO训练完成")
        print(f"  📊 平均奖励: {np.mean(episode_rewards):.3f}")
        print(f"  📊 最佳奖励: {max(episode_rewards):.3f}")

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'policy_losses': self.training_stats['policy_losses'],
            'value_losses': self.training_stats['value_losses']
        }

    def evaluate(
        self,
        env: TradingEnvironment,
        num_episodes: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """评估智能体"""
        print(f"📊 评估PPO智能体 ({num_episodes}轮)...")

        episode_results = []

        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_steps = 0

            while True:
                # 贪婪策略选择动作
                action, _, _ = self.select_action(state, training=False)

                # 执行动作
                state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                if terminated or truncated:
                    break

            episode_results.append({
                'episode': episode,
                'reward': episode_reward,
                'steps': episode_steps,
                'final_portfolio_value': info.get('portfolio_value', 0),
                'total_return': info.get('total_return', 0)
            })

            if verbose:
                print(f"  Episode {episode + 1}: 奖励={episode_reward:.2f}, "
                      f"投资组合=${info.get('portfolio_value', 0):,.2f}, "
                      f"收益率={info.get('total_return', 0):.2%}")

        # 计算统计信息
        rewards = [r['reward'] for r in episode_results]
        returns = [r['total_return'] for r in episode_results]

        evaluation_results = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'win_rate': len([r for r in returns if r > 0]) / len(returns),
            'best_return': max(returns),
            'worst_return': min(returns),
            'episode_results': episode_results
        }

        print(f"  📊 平均奖励: {evaluation_results['avg_reward']:.3f}")
        print(f"  📊 平均收益率: {evaluation_results['avg_return']:.2%}")
        print(f"  📊 胜率: {evaluation_results['win_rate']:.2%}")

        return evaluation_results

    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'n_symbols': self.n_symbols,
                'gamma': self.gamma,
                'eps_clip': self.eps_clip,
                'k_epochs': self.k_epochs,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef
            }
        }, filepath)
        print(f"💾 PPO模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']

        print(f"📂 PPO模型已从 {filepath} 加载")


class TradingRLAgent:
    """
    交易强化学习智能体管理器

    统一管理不同类型的RL智能体
    """

    def __init__(self):
        self.agents = {}
        self.training_history = {}

    def create_ppo_agent(
        self,
        name: str,
        state_dim: int,
        action_dim: int,
        n_symbols: int = 1,
        **kwargs
    ) -> PPOAgent:
        """创建PPO智能体"""
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            n_symbols=n_symbols,
            **kwargs
        )

        self.agents[name] = agent
        print(f"✅ 创建PPO智能体: {name}")

        return agent

    def train_agent(
        self,
        agent_name: str,
        env: TradingEnvironment,
        **kwargs
    ) -> Dict[str, Any]:
        """训练指定智能体"""
        if agent_name not in self.agents:
            raise ValueError(f"智能体 {agent_name} 不存在")

        agent = self.agents[agent_name]
        results = agent.train_on_environment(env, **kwargs)

        self.training_history[agent_name] = results

        return results

    def evaluate_agent(
        self,
        agent_name: str,
        env: TradingEnvironment,
        **kwargs
    ) -> Dict[str, Any]:
        """评估指定智能体"""
        if agent_name not in self.agents:
            raise ValueError(f"智能体 {agent_name} 不存在")

        agent = self.agents[agent_name]
        return agent.evaluate(env, **kwargs)

    def compare_agents(
        self,
        env: TradingEnvironment,
        num_episodes: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """比较所有智能体的性能"""
        print(f"🏆 比较智能体性能...")

        comparison_results = {}

        for agent_name, agent in self.agents.items():
            print(f"\n评估 {agent_name}:")
            results = agent.evaluate(env, num_episodes, verbose=False)
            comparison_results[agent_name] = results

        # 显示比较结果
        print(f"\n📊 性能比较:")
        print(f"{'智能体':<15} {'平均收益率':<12} {'胜率':<8} {'夏普比率':<10}")
        print("-" * 50)

        for agent_name, results in comparison_results.items():
            avg_return = results['avg_return']
            win_rate = results['win_rate']

            # 计算夏普比率（简化版）
            returns = [r['total_return'] for r in results['episode_results']]
            if len(returns) > 1:
                sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe = 0

            print(f"{agent_name:<15} {avg_return:<12.2%} {win_rate:<8.2%} {sharpe:<10.3f}")

        return comparison_results


# 便捷函数
def create_ppo_agent(
    state_dim: int,
    action_dim: int,
    n_symbols: int = 1,
    **kwargs
) -> PPOAgent:
    """创建PPO智能体"""
    return PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        n_symbols=n_symbols,
        **kwargs
    )


def create_trading_rl_manager() -> TradingRLAgent:
    """创建交易RL智能体管理器"""
    return TradingRLAgent()


# 测试代码
if __name__ == "__main__":
    print("🧠 强化学习智能体测试")
    print("=" * 50)

    # 创建交易环境
    env = create_trading_environment(["AAPL", "MSFT"])

    state_dim = env.observation_space.shape[0]
    action_dim = 3  # 0=卖出, 1=持有, 2=买入
    n_symbols = len(env.symbols)

    print(f"环境信息:")
    print(f"  状态维度: {state_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  股票数量: {n_symbols}")

    # 测试PPO智能体
    print("\n1. 测试PPO智能体:")
    ppo_agent = create_ppo_agent(
        state_dim=state_dim,
        action_dim=action_dim * n_symbols,  # 每只股票都有3种动作
        n_symbols=n_symbols
    )

    # 短期训练测试
    print("\n2. 短期训练测试:")
    training_results = ppo_agent.train_on_environment(
        env,
        num_episodes=10,
        update_frequency=5
    )

    # 评估测试
    print("\n3. 评估测试:")
    evaluation_results = ppo_agent.evaluate(env, num_episodes=3)

    # 测试RL管理器
    print("\n4. 测试RL管理器:")
    rl_manager = create_trading_rl_manager()

    # 创建多个智能体
    agent1 = rl_manager.create_ppo_agent(
        "PPO_v1", state_dim, action_dim * n_symbols, n_symbols
    )

    print("\n✅ 强化学习智能体测试完成！")