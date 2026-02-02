import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection
import seaborn as sns
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体支持（如系统无中文字体可注释掉）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False



class TerrainLossLandscape:


    def __init__(self, n_peaks=5, n_fissures=3, random_seed=42):
        np.random.seed(random_seed)
        self.n_peaks = n_peaks
        self.n_fissures = n_fissures

        # 生成山峰位置（局部最优解）
        self.peaks = np.random.randn(n_peaks, 2) * 3
        self.heights = np.random.rand(n_peaks) * 2 + 1
        self.widths = np.random.rand(n_peaks) * 0.5 + 0.5

        # 生成裂缝位置（梯度不连续区域）
        self.fissure_lines = []
        for _ in range(n_fissures):
            point = np.random.randn(2) * 2
            angle = np.random.rand() * 2 * np.pi
            direction = np.array([np.cos(angle), np.sin(angle)])
            self.fissure_lines.append((point, direction))

        self.base_curvature = 0.1

    def get_height(self, x, y):

        pos = np.array([x, y])
        height = self.base_curvature * (x ** 2 + y ** 2)

        # 添加负高斯峰（低谷）
        for peak, h, width in zip(self.peaks, self.heights, self.widths):
            dist = np.linalg.norm(pos - peak)
            height -= h * np.exp(-dist ** 2 / (2 * width ** 2))

        # 添加裂缝扰动
        for fissure_point, direction in self.fissure_lines:
            vec = pos - fissure_point
            perp = np.array([-direction[1], direction[0]])
            dist_to_fissure = abs(np.dot(vec, perp))

            if dist_to_fissure < 0.5:
                along = np.dot(vec, direction)
                zigzag = 0.2 * np.sin(along * 10) * (0.5 - dist_to_fissure)
                height += zigzag

        return height

    def get_gradient(self, x, y, eps=1e-5):

        dx = (self.get_height(x + eps, y) - self.get_height(x - eps, y)) / (2 * eps)
        dy = (self.get_height(x, y + eps) - self.get_height(x, y - eps)) / (2 * eps)
        return np.array([dx, dy])

    def get_curvature(self, x, y, eps=1e-5):

        h_center = self.get_height(x, y)
        h_xp = self.get_height(x + eps, y)
        h_xm = self.get_height(x - eps, y)
        h_yp = self.get_height(x, y + eps)
        h_ym = self.get_height(x, y - eps)

        d2x = (h_xp - 2 * h_center + h_xm) / (eps ** 2)
        d2y = (h_yp - 2 * h_center + h_ym) / (eps ** 2)
        return d2x + d2y

    def detect_fissure_type(self, x, y):

        height = self.get_height(x, y)
        grad = self.get_gradient(x, y)
        grad_norm = np.linalg.norm(grad)
        curvature = abs(self.get_curvature(x, y))

        if grad_norm < 0.2 and curvature > 1.0:
            return 1
        elif height > 2.0:
            return 2
        else:
            return 0


class CardiacMonitor:


    def __init__(self, window_size=50):
        self.window_size = window_size
        self.history = {
            'loss': [],
            'grad_norm': [],
            'learning_rate': [],
            'mode': [],
            'hessian_trace': [],
            'activation_mean': [],
            'activation_std': []
        }
        self.physiological_state = {
            'heart_rate': 0.01,
            'blood_pressure': 0,
            'rhythm_regularity': 1.0,
            'oxygen_level': 1.0
        }
        self.medication = []

    def examine(self, model: nn.Module, loss: float, iteration: int) -> Dict[str, any]:

        report = {}

        # 1. 测量血压（梯度范数）
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = np.sqrt(grad_norm)
        self.history['grad_norm'].append(grad_norm)

        # 2. 心电图分析（损失曲率）
        self.history['loss'].append(loss)
        if len(self.history['loss']) >= 3:
            curvature = abs(self.history['loss'][-1] - 2 * self.history['loss'][-2] +
                            self.history['loss'][-3])
        else:
            curvature = 0
        self.history['hessian_trace'].append(curvature)

        # 3. 病理诊断
        diagnosis = self._diagnose(grad_norm, curvature, loss)
        prescription = self._prescribe(diagnosis)

        return {
            'diagnosis': diagnosis,
            'prescription': prescription,
            'vitals': {
                'bp': grad_norm,
                'curvature': curvature,
                'hr': self.physiological_state['heart_rate']
            }
        }

    def _diagnose(self, grad_norm: float, curvature: float, loss: float) -> List[str]:

        diseases = []

        if grad_norm > 10.0:
            diseases.append('hypertension')
        if grad_norm < 1e-4 and loss > 0.1:
            diseases.append('hypotension')
        if len(self.history['loss']) > 10:
            recent_std = np.std(self.history['loss'][-10:])
            if recent_std > np.mean(self.history['loss'][-10:]) * 0.1:
                diseases.append('arrhythmia')
        if np.isnan(loss) or np.isinf(loss):
            diseases.append('cardiac_arrest')
        if curvature > 1.0:
            diseases.append('arteriosclerosis')

        return diseases

    def _prescribe(self, diseases: List[str]) -> Dict[str, float]:

        rx = {
            'lr_multiplier': 1.0,
            'liquid_ratio': 0.0,
            'regularization': 0.0,
            'defibrillation': False
        }

        for disease in diseases:
            if disease == 'hypertension':
                rx['lr_multiplier'] *= 0.5
                rx['liquid_ratio'] += 0.3
                self.medication.append(f'Iter {len(self.history["loss"])}: β-blocker (lr↓)')

            elif disease == 'hypotension':
                rx['lr_multiplier'] *= 2.0
                rx['liquid_ratio'] -= 0.1
                self.medication.append(f'Iter {len(self.history["loss"])}: Cardiac glycoside (lr↑)')

            elif disease == 'arrhythmia':
                rx['regularization'] += 0.01
                rx['liquid_ratio'] += 0.2
                self.medication.append(f'Iter {len(self.history["loss"])}: Anticoagulant (reg↑)')

            elif disease == 'arteriosclerosis':
                rx['liquid_ratio'] += 0.4
                self.medication.append(f'Iter {len(self.history["loss"])}: Vasodilator (complexity↓)')

            elif disease == 'cardiac_arrest':
                rx['defibrillation'] = True
                self.medication.append(f'Iter {len(self.history["loss"])}: DEFIBRILLATION (reset!)')

        return rx

class HeartbeatScheduler:


    def __init__(self, optimizer, base_lr=0.01, cardiac_cycle=100):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.cardiac_cycle = cardiac_cycle
        self.current_beat = 0
        self.emergency_mode = False

    def step(self, prescription: Dict[str, float], iteration: int):

        # 基础窦性心律（正弦调制）
        phase = 2 * np.pi * (iteration % self.cardiac_cycle) / self.cardiac_cycle
        sinus_rhythm = 1 + 0.2 * np.sin(phase)

        # 应用处方
        if prescription.get('defibrillation'):
            self._defibrillate()
            return self.base_lr * 0.1

        adjusted_lr = self.base_lr * sinus_rhythm * prescription.get('lr_multiplier', 1.0)
        adjusted_lr = np.clip(adjusted_lr, 1e-5, 0.5)

        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr

        return adjusted_lr

    def _defibrillate(self):

        self.optimizer.state = {}
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * 0.1
        self.emergency_mode = True
        print("[CARDIAC] ⚡ DEFIBRILLATION APPLIED! ⚡")


class FissureLiquidSolidWithHeart(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_fissures=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_fissures = n_fissures

        # 固态大地部分（骨骼）
        self.solid_skeleton = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # 液态血管网络（可动态调整管径）
        self.liquid_vessels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ) for _ in range(n_fissures)
        ])

        # 心脏门控：控制血流分配
        self.cardiac_gates = nn.Sequential(
            nn.Linear(hidden_dim, n_fissures),
            nn.Softmax(dim=-1)
        )

        # 时间常数（血液粘稠度）
        self.viscosity = nn.Parameter(torch.ones(n_fissures) * 0.5)

        # 输出层（运动神经）
        self.motor_neuron = nn.Linear(hidden_dim, output_dim)

    def liquid_dynamics(self, h, dt, input_signal, viscosity):
        """
        LTC微分方程数值解（Runge-Kutta 4）
        dh/dt = -h/τ + tanh(W*h + b)
        """

        def f(state):
            return -state / (viscosity + 0.1) + torch.tanh(state + 0.1 * input_signal)

        # RK4积分
        k1 = f(h)
        k2 = f(h + 0.5 * dt * k1)
        k3 = f(h + 0.5 * dt * k2)
        k4 = f(h + dt * k3)

        return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, x, heart_state: Optional[Dict[str, float]] = None, dt=0.1):

        batch_size = x.size(0)

        # 1. 固态骨骼阶段
        bone_signal = self.solid_skeleton(x)

        # 2. 心脏泵血分配（门控）
        blood_distribution = self.cardiac_gates(bone_signal)

        # 3. 液态流动（血管网络）
        liquid_outputs = []
        for i, vessel in enumerate(self.liquid_vessels):
            # 基础流动
            flow = vessel(bone_signal)

            # 应用粘稠度调节（由心脏处方控制）
            if heart_state and 'liquid_ratio' in heart_state:
                current_viscosity = self.viscosity[i] * (1 + heart_state['liquid_ratio'])
            else:
                current_viscosity = self.viscosity[i]

            # LTC动态（模拟血液流动阻力）
            flow = self.liquid_dynamics(flow, dt, bone_signal, current_viscosity)
            liquid_outputs.append(flow)

        # 4. 融合：骨骼支撑 + 血液营养
        liquid_stack = torch.stack(liquid_outputs, dim=1)
        blood_distribution = blood_distribution.unsqueeze(-1)

        nourished_signal = (blood_distribution * liquid_stack).sum(dim=1) + bone_signal

        output = self.motor_neuron(nourished_signal)
        return output, blood_distribution.squeeze()

class CardiacFissureTrainer:

    def __init__(self, model, terrain, lr=0.01):
        self.model = model
        self.terrain = terrain
        self.cardiac_monitor = CardiacMonitor()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.heart_scheduler = HeartbeatScheduler(self.optimizer, base_lr=lr)
        self.steel_ball_pos = np.array([4.0, 4.0])
        self.trajectory = [self.steel_ball_pos.copy()]

    def train_epoch(self, X, y, epoch):
        self.model.train()
        n_samples = len(X)
        batch_size = 32
        n_batches = n_samples // batch_size

        epoch_stats = {
            'losses': [],
            'grad_norms': [],
            'modes': [],
            'heart_rate': [],
            'positions': []
        }

        print(f"\n[Epoch {epoch}] 开始心脏周期...")

        for batch_idx in range(n_batches):
            # 数据采样（血液循环）
            idx = np.random.choice(n_samples, batch_size)
            batch_x = X[idx]
            batch_y = y[idx]

            # 默认处方
            prescription = {'liquid_ratio': 0.0, 'lr_multiplier': 1.0, 'defibrillation': False}

            # 每10步体检一次（心脏听诊）
            if batch_idx > 0 and batch_idx % 10 == 0:
                recent_loss = np.mean(epoch_stats['losses'][-10:]) if epoch_stats['losses'] else 1.0
                diagnosis = self.cardiac_monitor.examine(self.model, recent_loss,
                                                         epoch * n_batches + batch_idx)
                prescription = diagnosis['prescription']

                if diagnosis['diagnosis']:
                    print(f"  [Iter {batch_idx}] 诊断: {diagnosis['diagnosis']}, "
                          f"处方: lr×{prescription['lr_multiplier']:.2f}, "
                          f"液态比:{prescription['liquid_ratio']:.2f}")

            # 心脏调控学习率（窦性心律 + 病理调节）
            current_lr = self.heart_scheduler.step(prescription, epoch * n_batches + batch_idx)
            epoch_stats['heart_rate'].append(current_lr)

            # 前向传播（考虑当前生理状态）
            self.optimizer.zero_grad()
            outputs, blood_dist = self.model(batch_x, prescription)
            loss = nn.MSELoss()(outputs.squeeze(), batch_y)

            # 反向传播（静脉回流）
            loss.backward()

            # 病理干预（除颤等）
            if prescription.get('defibrillation'):
                self._reset_liquid_vessels()

            # 更新参数（心脏泵血推动钢珠）
            self.optimizer.step()

            # 记录钢珠运动（在地形中的轨迹）
            grad = self.terrain.get_gradient(self.steel_ball_pos[0], self.steel_ball_pos[1])
            grad_norm = np.linalg.norm(grad)

            # 钢珠运动方程：受学习率（心率）和地形梯度共同驱动
            movement = -0.1 * grad * prescription['lr_multiplier']
            # 添加裂缝随机扰动（如果在液态模式）
            if prescription['liquid_ratio'] > 0.2:
                movement += np.random.randn(2) * 0.05 * prescription['liquid_ratio']

            self.steel_ball_pos += movement
            self.steel_ball_pos = np.clip(self.steel_ball_pos, -5, 5)
            self.trajectory.append(self.steel_ball_pos.copy())

            # 记录统计
            epoch_stats['losses'].append(loss.item())
            epoch_stats['grad_norms'].append(grad_norm)
            epoch_stats['modes'].append(1 if prescription['liquid_ratio'] > 0.2 else 0)
            epoch_stats['positions'].append(self.steel_ball_pos.copy())

        return epoch_stats

    def _reset_liquid_vessels(self):
        with torch.no_grad():
            for vessel in self.model.liquid_vessels:
                for layer in vessel:
                    if isinstance(layer, nn.Linear):
                        layer.weight.normal_(0, 0.01)
                        layer.bias.zero_()
        print("[CARDIAC] Liquid vessels reset completed.")

def visualize_cardiac_training(trainer, all_stats, terrain, n_epochs):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :2])
    losses = []
    for stats in all_stats:
        losses.extend(stats['losses'])

    t = np.arange(len(losses))
    ax1.plot(t, losses, 'darkred', linewidth=1.5, label='Loss Signal')
    if len(losses) > 10:
        peaks, _ = find_peaks([-l for l in losses], distance=20, prominence=0.1)
        ax1.scatter(peaks, [losses[p] for p in peaks], c='red', s=100,
                    marker='v', zorder=5, label='R-peaks (Breakthrough)')

    ax1.set_title('ECG: Training Loss (心电图：损失信号)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Amplitude (Loss)')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend()
    ax2 = fig.add_subplot(gs[1, :2])
    grads = []
    for stats in all_stats:
        grads.extend(stats['grad_norms'])

    ax2.plot(t, grads, 'blue', linewidth=2, label='Gradient Pressure')
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Hypertension Threshold')
    ax2.fill_between(t, 0, grads, alpha=0.3, color='blue')
    ax2.set_title('Blood Pressure: Gradient Norm (血压：梯度压力)', fontsize=14)
    ax2.set_ylabel('Pressure (||∇L||)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax3 = fig.add_subplot(gs[2, :2])
    hrs = []
    for stats in all_stats:
        hrs.extend(stats['heart_rate'])
    ax3.plot(t, hrs, 'green', linewidth=2, label='Heart Rate (LR)')
    ax3.set_title('Heart Rate: Learning Rate Schedule (心率：学习率)', fontsize=14)
    ax3.set_xlabel('Time (Iterations)')
    ax3.set_ylabel('Rate (lr)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax4 = fig.add_subplot(gs[0, 2])
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[terrain.get_height(xi, yi) for xi in x] for yi in y])

    contour = ax4.contourf(X, Y, Z, levels=20, cmap='terrain', alpha=0.6)
    ax4.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    traj = np.array(trainer.trajectory)
    scatter = ax4.scatter(traj[:, 0], traj[:, 1], c=range(len(traj)),
                          cmap='hot', s=20, alpha=0.8)
    ax4.plot(traj[:, 0], traj[:, 1], 'k--', alpha=0.3, linewidth=1)
    ax4.scatter(traj[0, 0], traj[0, 1], c='green', s=200, marker='o',
                edgecolors='black', label='Start', zorder=5)
    ax4.scatter(traj[-1, 0], traj[-1, 1], c='red', s=200, marker='*',
                edgecolors='black', label='End', zorder=5)
    plt.colorbar(scatter, ax=ax4, label='Time')
    ax4.set_title('Steel Ball in Terrain (钢珠运动)', fontsize=12)
    ax4.legend()

    ax5 = fig.add_subplot(gs[1:, 2])
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.axis('off')
    ax5.set_title('Anatomy: FLSHS Structure (解剖结构)', fontsize=14, fontweight='bold')


    heart = Ellipse((5, 8), 2, 1.5, color='red', alpha=0.6)
    ax5.add_patch(heart)
    ax5.text(5, 8, 'Heart\n(Monitor)', ha='center', va='center',
             fontsize=10, fontweight='bold', color='white')
    ax5.add_patch(FancyBboxPatch((2, 5.5), 6, 1, boxstyle="round,pad=0.1",
                                 color='gray', alpha=0.5))
    ax5.text(5, 6, 'Solid Skeleton\n(固态骨骼)', ha='center', fontsize=9)
    for i in range(3):
        y_pos = 4.0 - i * 0.8
        ax5.plot([5, 5], [5.5, y_pos], 'b-', alpha=0.4, linewidth=3)
        circle = Circle((5, y_pos), 0.3, color='cyan', alpha=0.6)
        ax5.add_patch(circle)
    ax5.text(7.5, 2.5, 'Liquid Vessels\n(液态血管)', fontsize=9)
    ball = Circle((5, 1), 0.4, color='silver', ec='black', linewidth=2)
    ax5.add_patch(ball)
    ax5.text(5, 1, 'Steel Ball\n(Params)', ha='center', va='center', fontsize=8)
    ax5.arrow(5, 7.2, 0, -0.5, head_width=0.3, head_length=0.2, fc='red', ec='red')
    ax5.arrow(5, 3.5, 0, -1.8, head_width=0.3, head_length=0.2, fc='blue', ec='blue')

    plt.suptitle('Cardiac-Fissure-Liquid-Solid Training Monitor (生命体征监测)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('cardiac_training_complete.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n" + "=" * 60)
    print("CARDIAC DIAGNOSIS REPORT (心脏诊断报告)")
    print("=" * 60)
    if trainer.cardiac_monitor.medication:
        for med in trainer.cardiac_monitor.medication:
            print(f"  {med}")
    else:
        print("  无异常 - 生理指标正常")
    print("=" * 60)
def demo_cardiac_training():
    print("=" * 70)
    print("初始化心脏-钢珠-裂隙生命体学习系统")
    print("=" * 70)
    print("系统组成：")
    print("1. 大地 (Terrain): 复杂损失景观，含山峰与裂缝")
    print("2. 钢珠 (Steel Ball): 参数向量，在地形中滚动")
    print("3. 心脏 (Heart): 生理监测与自动药物干预")
    print("4. 液态血管 (Liquid): LTC动态网络，应对裂缝")
    print("5. 固态骨骼 (Solid): 稳定基础特征提取")
    print("=" * 70)

    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 生成复杂地形数据
    X = torch.linspace(-3, 3, 500).unsqueeze(1)
    y = torch.sin(X * 2) * torch.cos(X * 0.5) + 0.1 * torch.randn(500, 1)
    y = y.squeeze()

    # 创建地形
    terrain = TerrainLossLandscape(n_peaks=6, n_fissures=3)

    # 创建心脏增强模型
    model = FissureLiquidSolidWithHeart(input_dim=1, hidden_dim=32, output_dim=1, n_fissures=3)
    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters())}")

    # 创建训练器
    trainer = CardiacFissureTrainer(model, terrain, lr=0.02)

    print("\n开始生命体训练...")
    print("监测指标：ECG(损失), BP(梯度), HR(学习率)")
    print("干预措施：β阻滞剂(降lr)/强心剂(增lr)/抗凝剂(正则)/除颤(重置)")
    print("=" * 70)

    all_stats = []
    n_epochs = 5

    for epoch in range(n_epochs):
        stats = trainer.train_epoch(X, y, epoch)
        all_stats.append(stats)
        avg_loss = np.mean(stats['losses'])
        avg_hr = np.mean(stats['heart_rate'])
        n_liquid = sum(stats['modes'])
        print(f"[Epoch {epoch} 完成] 平均损失: {avg_loss:.4f}, "
              f"平均心率: {avg_hr:.5f}, 液态模式比例: {n_liquid / len(stats['modes']):.1%}")


    visualize_cardiac_training(trainer, all_stats, terrain, n_epochs)
    model.eval()
    with torch.no_grad():
        X_test = torch.linspace(-3, 3, 100).unsqueeze(1)
        y_pred, blood_dist = model(X_test)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X.numpy(), y.numpy(), alpha=0.3, label='Data (Terrain)', s=10, c='gray')
    plt.plot(X_test.numpy(), y_pred.numpy(), 'r-', linewidth=2, label='Model Prediction')
    plt.title('Final Model Fit (最终拟合效果)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(blood_dist.numpy())
    plt.title('Blood Distribution (心脏血流分配)')
    plt.xlabel('Sample')
    plt.ylabel('Gate Value')
    plt.legend([f'Vessel {i + 1}' for i in range(blood_dist.size(1))])
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('final_results.png', dpi=150)
    plt.show()

    print("\n训练完成！可视化结果已保存为 'cardiac_training_complete.png' 和 'final_results.png'")


if __name__ == "__main__":
    demo_cardiac_training()