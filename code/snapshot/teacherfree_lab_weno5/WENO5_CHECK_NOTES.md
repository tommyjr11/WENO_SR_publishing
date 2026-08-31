# teacherfree_lab_weno5 检查报告（2026-07-08）

对照对象：`weno5_rk3_diff.py`（部署的 WENO5-MLP warp kernel）、
`warp_weno5_helpers.py`（classical/gauss kernel）、`pretrain_weno5_offline.py`
（正典 torch 定义）。结论：**移植正确，只做了 2 处小修改（见 §2）**。
`bash teacherfree_lab_weno5/run_weno5_v1.sh` 可以直接开跑。

## 1. 逐项核对结果（全部一致，未改动）

| 项目 | 对照 | 结果 |
|---|---|---|
| 5 维特征 | `weno5_rk3_diff.weno5_nn_features` | 逐式一致：delta 传感器（13/12·abs(d2)+1/4·abs(...)，abs 版非平方版）、gamma_s（clamp 0..1，eps 1e-15）、q_scale clamp min=1、scale_feature=(log10+16)/16 clamp 0..1 |
| MLP 前向 | 同上 kernel | swish×3、badness=6·tanh(raw/6)、softmax（warp 减 max 数值稳定版=同一函数）、β=3r、α=d/(β+1e-12)²、归一 |
| 线性初始化 | 部署 meta 约定 | w4=b4=0 → r=[1/3,1/3,1/3] → ω≡d，已数值验证到 1e-12 |
| 界面系数 lr=1/2 | `stencil_2d` loc 1-3 / 2-4 | 一致，d=(0.1,0.6,0.3)/(0.3,0.6,0.1) |
| gauss 系数 lr=3/4 | `stencil_gauss` loc 1-3 / 4-6 | 数值一致（−√3/12、√3/3、1−√3/4 等），d=((210±√3)/1080, 11/18, ...) 配对正确；lr3↔x=−√3/6、lr4↔+√3/6 |
| SSPRK3 | 标准 Shu-Osher | u1 = u+dtL(u); u2 = 3/4u+1/4(u1+dtL(u1)); u = 1/3u+2/3(u2+dtL(u2)) ✓ |
| 上风通量/ghost | WENO7 版同构 | ghost=2、窗口对齐 i+1/2、roll(1) 差分 ✓ |
| plateau 回退 d | `plateau_detected`（1e-13·q_scale） | 阈值与部署一致 |
| checkpoint 格式 | `load_mlp_params` | (1,5,10)... 形状与 meta 架构串（含续训所需子串）均匹配 |
| eno-cutoff | validate 脚本 `--no-eno-cutoff` | 与训练（无 cutoff）一致 |
| 门带 1e-7~1e-3 | classical ε 推导 | **不用改**：scale_feature 公式同 WENO7，ε=1e-6、β~振幅² ⇒ 亚可分辨阈值 √ε≈1e-3，与 WENO7 同理 |

## 2. 修改了什么（共 2 处）

1. **`weno5_core.py` 新增 `check_weno5_coefficients()`，训练启动时自动运行**
   （`train_apost_weno5.run()` 第一行调用）。用单项式 x^k 的精确单元平均做
   第一性原理自检：每个 3 格子模板必须对 deg≤2 多项式在目标点精确重构
   （3 阶候选），d 加权组合必须对 deg≤4 精确（5 阶线性格式），四个目标点
   x=+1/2、−1/2、±√3/6 全查，容差 1e-12。**系数错一个数字就断言失败**——
   这正面回答了"WENO5 系数我没算"：代码每次启动都替你算。已实测通过。
2. **probe/EVAL 改回 L2**（原来传了 `err_power=4`）。训练 loss 仍是 L4 不变；
   只有监控用的 vs_cls 比值换回 L2，这样数字才能和 WENO7 那边的历史监控
   （sharp≈0.6~0.7 一类经验值）直接对比。

## 3. 冒烟测试（已通过）

3 步端到端（训练+eval+checkpoint）无错。启动标定实测：

- `bound_viol: LINEAR=5.5e-3 CLASSICAL=2.3e-4` → `--bound-floor 2e-4`
  正好压在 classical 水平（与 WENO7 版的设定逻辑相同）；
- `ampgate: CLASSICAL=8.3e-3`（WENO5 的 ε 在门带内比 WENO7 更接近 d，
  floor 1e-3 略严于 classical，安全侧）；
- step1 `kl_ag=0.000e+00` → 线性初始 ω≡d 的又一验证。

## 4. 2026-07-08 追加修正

- **训练内置旧 Warp RK3 二维 Sod 健康检查**：每个 `--eval-interval`（默认 250 步）
  会把当前 Torch MLP checkpoint 临时转成旧 runner 的 `wp.array` 权重，然后调用
  `run_weno5_circle_mlp_compare.advance_one_step` 跑一个可信二维 planar Sod 小算例。
  默认是 `100x8, t=0.25, CFL=0.4, exact-sod, axis=x, characteristic,
  ENO cutoff on`，输出到
  `teacherfree_lab_weno5/runs/.../circle_validation/step_000250/`：
  `density_compare.png`、`density_difference.png`、`centerline_density.png`、
  `summary.txt`、`circle_sod_results.npz`。总表在 run 根目录：
  `circle_validation_metrics.csv/.npz` 和 `circle_validation_trends.png`。
  这不是自写 1D Sod solver；空间离散和时间推进走的就是旧 WENO5 Warp RK3 前向路径。
- **大图验证仍走旧 quadrant runner**：`validate_weno5_quadrant.sh` 是新 checkpoint
  接入旧 `run_weno5_quadrant_mlp_only.py` 的包装脚本；默认复刻旧图目录
  `plots/WENO5_MLP/weno5_quadrant_400_t05_5_10_6_6_3_step137000_mlp_only_evilin_cuda`
  的设置：400x400、t=0.5、CFL=0.4、case12、evilin、characteristic、no ENO cutoff。
  第一次运行可能需要 Warp 编译，等 10 分钟左右是正常的。
- **smooth anchor 改为 canonical 生成器**：原先的手写多项式+微弱正弦 stencil
  已替换为 WENO5 canonical smooth families 的 exact cell-average 生成器
  （sine/tanh/cubic/quadratic/linear/constant），对应 WENO7 版 canonical anchor
  的做法。

## 5. v1 结果分析与 v2 设计（2026-07-08 晚）

v1（tv 0.03、门带 1e-7~1e-3、floor 1e-3）：Sod gain 峰值 33.8%@2k，
随后单调衰减到 13.5%@12k（WENO5 本身耗散重，tv_bg 的回拉比 WENO7 明显）；
case6@4000/5500 弱弧区有锯齿。

**实测定位（关键数据）**：对 case6@4000 的密度场取 y=0.25/0.30/0.35 三条
水平割线（x<0.48，避开竖直激波），锯齿二阶差分幅值 = **1.4~1.5e-3 相对量**
——恰好落在 v1 门带上限（amp-max=1e-3）**外侧一点点**，门控没罩住它。

结论：门带必须**扩宽**（amp-max → 3e-3），不能缩窄（1e-4 方向相反，
会把 1e-4~1e-3 段也放开）。"WENO5 模板少所以震荡引入少"不成立：
3 格 3 阶子模板在权重给错时相对外插误差更大，实测同门控下比 WENO7 更振。
另：floor 收紧（1e-3→1e-4）与门带缩窄是两个相反方向的改动，不要混在一起。

v2（`run_weno5_v2.sh`，双卡单变量对照，checkpoint/eval 加密到 200 步）：

- GPU0 `v2a_gate3e3`：只扩门带到 3e-3，tv 保持 0.03（保 gain，单变量）；
- GPU1 `v2b_gate3e3_f1e4_tv0038`：门带 3e-3 + floor 1e-4 + tv 0.038
  （更重的抛光组合，验证强压是否伤 gain）。

判据：case6 弱弧锯齿消失（用同样的割线二阶差分复测，应回到 classical
水平）且 Sod gain 峰值段（2~4k 步）保持 ≥25%。v2a 干净就取 v2a
（gain 更高）；仍有残余才用 v2b。

## 6. 为什么 WENO7 用 1e-3 门带就够、WENO5 不够（2026-07-08 深夜，实测定论）

用同一族 sigmoid 前沿 stencil（底值 O(1)，只变相对幅值 a）测 classical 权重
相对线性权重 d 的平均 KL（对所有 lr 目标点平均）：

| 相对幅值 a | KL cls WENO5 | KL cls WENO7 |
|---|---|---|
| 1e-4   | 1.4e-7 | 2.6e-7 |
| 3e-4   | 1.2e-5 | 1.9e-5 |
| 1e-3   | 1.1e-3 | 1.4e-3 |
| 1.5e-3 | 4.0e-3 | 5.7e-3 |
| 3e-3   | 2.0e-2 | 2.2e-2 |
| 1e-2   | 6.3e-2 | 5.5e-2 |

三条结论：

1. **两个格式的 ε 线性化边界几乎重合**（都在 ~1e-3 开始离开线性，WENO7 甚至略早）。
   "WENO5 classical 阈值天生更宽所以要更宽门带"的说法**不成立**。
2. 差别在**模型侧**：门带只钉住 [amp-min, amp-max] 内的行为，带外由 rollout loss
   决定。WENO5 底子耗散更重（v1 也观察到 gain 衰减更快），rollout 为对抗耗散把
   反耗散补偿推得更狠，过陡带上沿伸到 1.4~1.5e-3，越过 1e-3 门沿；WENO7 重构
   本身更准、补偿更温和，残余过陡缩在 1e-3 内，所以同门带就干净。
3. classical 在 1.5e-3 处 KL 也非零（4e-3），即 classical 那里同样非线性——但它
   偏向**加耗散**方向所以不出锯齿；MLP 偏向**反耗散**才出锯齿。门带治的是方向。

可写进论文的定门带准则：**amp-max 不是普适常数，而是"实测该格式训练后过陡带
上沿 + 余量"**，用 2D 密度场割线的二阶差分幅值即可测（§5 的方法）。
"多训几步就好"是反的——WENO7 v18 的教训是练得越久小振幅过陡越重。

## 7. 已知差异 / 注意事项（无需改代码）

- 选型依据 = eval.csv 的 vs_cls 五族比值 + 内置 `circle_validation` 100x8 Sod
  健康线 + `validate_weno5_quadrant.sh` 跑旧 Warp RK3 2D 大图。
  经验规律直接搬用 WENO7 的：甜点大概率在 vs_cls_sharp 明显 <1 且训练早中期
  （几千步）的检查点。
- WENO7 的教训在此同样适用：**练得越久小振幅越易被过度压陡**；
  本版已带 ampgate（λ=1.0、门带 1e-7~1e-3、floor 1e-3），理论上长训不退化，
  但请按 v19 的验法在 ~2k 和 ~10k+ 各取检查点跑 2D 对比确认。
- validate 脚本默认 case12、t_end=0.5；跑 case6 记得 `CASE=case6 T_END=0.25`。
- smooth 锚样本已经改成 canonical exact cell-average 生成器，和 WENO7 的做法一致。
