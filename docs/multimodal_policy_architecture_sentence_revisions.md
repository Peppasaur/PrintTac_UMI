# Multimodal Policy Architecture 逐句修改说明

本文档根据当前 `reactive_diffusion_policy` 实现，对论文中的 **Multimodal Policy Architecture** 和 **Action Representation and Closed-Loop Execution** 两个小节进行逐句核对。

每项均按以下格式整理：

1. 原句；
2. 建议修改结果；
3. 修改原因。

文末提供一份可以直接复制到论文中的完整 LaTeX 版本。

## 快速结论

原文的总体框架基本正确：系统使用 ResNet-18、低维磁触觉与本体状态、条件 1D U-Net、相对末端动作和 receding-horizon 执行。

需要重点修正的地方包括：

- ResNet-18 使用标准全局平均池化，不使用 SpatialSoftmax。
- 多模态历史被展平为全局条件向量，而不是具有注意力语义的 token。
- 1D U-Net 使用 FiLM 条件调制，但没有 cross-attention。
- 标准策略使用两步视觉、触觉和本体状态历史。
- first-frame 策略只持续更新触觉；TCP 和夹爪状态不作为持续变化的网络输入。
- 动作为 10 维：3D 平移、6D 连续旋转表示和 1D 绝对夹爪命令。
- 实际闭环执行使用异步推理和 temporal ensemble，不是简单执行固定前 $k$ 步后丢弃其余预测。

---

## 1. Figure caption

### 原句

```latex
\caption{\textbf{Pipeline of the multimodal conditional diffusion policy.} Visual frames from the wrist camera are encoded via a ResNet-18 backbone, while normalized low-dimensional tactile vectors and proprioceptive states are concatenated directly without a dedicated encoder. The unified multimodal embedding conditions a 1D convolutional U-Net to denoise action trajectories in a receding-horizon fashion.}
```

### 修改后

```latex
\caption{\textbf{Pipeline of the multimodal conditional diffusion policy.}
Wrist-camera images are encoded by a ResNet-18 backbone with global average pooling, while normalized low-dimensional tactile and proprioceptive vectors are concatenated directly without dedicated encoders. Features from the observation history are flattened into a global conditioning vector, which modulates a 1D convolutional U-Net through feature-wise linear modulation (FiLM) to denoise future action trajectories.}
```

### 修改原因

- ResNet-18 保留标准全局平均池化。
- 多模态条件在代码中是一个展平后的向量。
- 条件通过 FiLM 注入 U-Net；代码中没有 cross-attention。

---

## 2. 网络结构总述

### 原句

```latex
The policy architecture adopts a conditional 1D convolutional U-Net that denoises an action sequence conditioned on an observation history window (Fig.\~\ref{fig:policy_architecture}).
```

### 修改后

```latex
The policy employs a conditional 1D convolutional U-Net that denoises an action sequence conditioned on a fixed-length multimodal observation history (Fig.~\ref{fig:policy_architecture}).
```

### 修改原因

- 修正 `Fig.\~` 的 LaTeX 写法。
- 明确条件输入是固定长度的多模态历史。

---

## 3. Visual Observations

### 原句

```latex
\textit{Visual Observations}: An observation history $\mathbf{O}*t = {o*{t-1}, o_t}$ comprising the two most recent RGB frames captured by the wrist-mounted camera. Each frame is processed by a ResNet-18 visual backbone~\cite{he2016deep} to extract a compact visual embedding.
```

### 修改后

```latex
\item \textit{Visual Observations}: The standard closed-loop policy receives an observation history
$\mathbf{O}_t=\{o_{t-1},o_t\}$ containing the two most recent RGB frames captured by the wrist-mounted camera. Each frame is processed independently by a ResNet-18 backbone~\cite{he2016deep}, whose global average pooling layer produces a 512-dimensional visual feature.
```

### 修改原因

- 修正公式中的下标和集合格式。
- 两个时间步的图像分别经过同一个类型的 ResNet-18 编码。
- 当前 ResNet-18 将最后的全连接层替换为 Identity，但保留全局平均池化，因此每帧输出 512 维特征。
- 当前视觉编码器不使用 SpatialSoftmax。

---

## 4. Tactile Observations：输入形式

### 原句

```latex
\textit{Tactile Observations}: A temporal window of multi-axis magnetic flux measurements $\mathbf{B}*t = {b*{t-1}, b_t}$, where each $t_\tau$ concatenates the 3-axis readings across the magnetometer channels of the tactile finger.
```

### 单传感器版本

```latex
\item \textit{Tactile Observations}: The policy receives a temporal window of magnetic tactile measurements
$\mathbf{B}_t=\{\mathbf{b}_{t-1},\mathbf{b}_t\}$. Each vector $\mathbf{b}_{\tau}$ is obtained by temporally averaging the baseline-subtracted three-axis readings from the magnetometer channels and flattening them into a low-dimensional tactile vector.
```

### 双传感器版本

如果论文描述的是当前双磁传感器设置，建议使用下面这一版：

```latex
\item \textit{Tactile Observations}: The policy receives temporal measurements from two magnetic tactile sensors. For sensor $j\in\{1,2\}$, the tactile history is
$\mathbf{B}^{(j)}_t=\{\mathbf{b}^{(j)}_{t-1},\mathbf{b}^{(j)}_t\}$.
Each $\mathbf{b}^{(j)}_{\tau}$ is obtained by temporally averaging the baseline-subtracted three-axis readings from its magnetometer channels and flattening them into a 15-dimensional tactile vector. The two sensor vectors are normalized independently and concatenated as
$\mathbf{b}_{\tau}=[\mathbf{b}^{(1)}_{\tau},\mathbf{b}^{(2)}_{\tau}]$.
```

### 修改原因

- 原文中的 `$t_\tau$` 是笔误，应为 `$\mathbf{b}_\tau$`。
- 真实机器人环境会对一个观测帧内收到的磁传感器样本取时间平均。
- 双传感器配置使用两个独立的 15 维触觉输入，并分别使用训练集统计量归一化。

---

## 5. Tactile Observations：编码与归一化

### 原句

```latex
Crucially, unlike vision-based tactile approaches that require complex pre-trained encoders (e.g., tactile ViTs or CNNs), $\mathbf{B}_t$ is normalized with training-set statistics and concatenated directly into the state vector, without a dedicated tactile encoder.
```

### 修改后

```latex
Unlike vision-based tactile approaches that employ pretrained tactile CNNs or vision transformers, each tactile vector is standardized using statistics computed from the training set and is concatenated directly with the other observation features, without a dedicated tactile encoder.
```

### 修改原因

- 当前实现对每个触觉输入向量进行标准化。
- 标准化后的低维向量直接与视觉和本体特征拼接，不经过单独的 CNN、ViT 或 MLP 触觉编码器。

---

## 6. Proprioceptive State

### 原句

```latex
\item \textit{Proprioceptive State}: The robot's current proprioceptive state $\mathbf{P}_t$, including continuous gripper aperture.
```

### 修改后

```latex
\item \textit{Proprioceptive State}: The standard policy receives a proprioceptive history
$\mathbf{P}_t=\{\mathbf{p}_{t-1},\mathbf{p}_t\}$. Each state $\mathbf{p}_{\tau}$ contains the end-effector position, a continuous 6D rotation representation, and the continuous gripper state. When relative-action training is enabled, the end-effector poses in the observation window are expressed relative to the most recent observed pose.
```

### 修改原因

- 标准策略输入两步本体状态，而不只是当前一步。
- TCP 位姿由 3D 位置和 6D 连续旋转表示组成。
- `relative_action=True` 时，TCP observation 也会转换为相对于最新观测 TCP 的表示。

---

## 7. Global conditioning

### 原句

```latex
The flattened visual embeddings, normalized tactile readings, and proprioceptive features are concatenated into a global conditioning token $\mathbf{c}*t = [\text{Enc}(o*{t-1}, o_t), \mathbf{T}_t, \mathbf{P}_t]$, which modulates the intermediate features of the 1D U-Net via Feature-wise Linear Modulation~\cite{perez2018film} and cross-attention layers.
```

### 修改后

```latex
For each observation step $\tau$, the visual, tactile, and proprioceptive features are concatenated as
\[
\mathbf{z}_{\tau}
=
[\operatorname{ResNet18}(o_{\tau}),
 \tilde{\mathbf{b}}_{\tau},
 \tilde{\mathbf{p}}_{\tau}],
\]
where the tilde denotes normalization. The features from the two observation steps are then flattened into a single global conditioning vector,
\[
\mathbf{c}_t
=
\operatorname{Flatten}
(\mathbf{z}_{t-1},\mathbf{z}_t).
\]
This conditioning vector is combined with the diffusion-timestep embedding and modulates the residual blocks of the 1D U-Net through feature-wise linear modulation (FiLM)~\cite{perez2018film}. The architecture does not employ cross-attention.
```

### 修改原因

- 应称为 global conditioning vector，而不是 token。
- 实际实现先构造每个时间步的多模态特征，再将两个时间步展平。
- diffusion timestep embedding 与多模态全局条件进行拼接。
- 当前 1D U-Net 中没有 cross-attention 层。

---

## 8. Future action sequence

### 原句

```latex
Following the UMI formulation~\cite{chi2024universal}, the policy predicts a sequence of $n$ future end-effector waypoints $\mathbf{A}*t = {a*{t+1}, \dots, a_{t+n}}$.
```

### 修改后

```latex
Following the relative-action formulation used in UMI~\cite{chi2024universal}, the policy predicts a sequence of $n$ future end-effector waypoints,
\[
\mathbf{A}_t
=
\{a_{t+1},\ldots,a_{t+n}\}.
\]
```

### 修改原因

- 修正公式格式。
- 明确当前方法借鉴的是 UMI 的相对动作表示。

---

## 9. Action waypoint representation

### 原句

```latex
Each action waypoint $a_\tau = (\Delta \mathbf{T}*\tau, \mathbf{w}*\tau)$ consists of a 6-DoF end-effector pose relative to the current tool frame along with the absolute gripper opening width $\mathbf{w}_\tau \in [0, 1]$.
```

### 修改后

```latex
Each action waypoint is represented as
\[
a_{\tau}
=
[\Delta\mathbf{p}_{\tau},
 \mathbf{r}^{6D}_{\tau},
 w_{\tau}]
\in\mathbb{R}^{10},
\]
where $\Delta\mathbf{p}_{\tau}\in\mathbb{R}^{3}$ and
$\mathbf{r}^{6D}_{\tau}\in\mathbb{R}^{6}$ represent the relative end-effector transform
\[
\Delta\mathbf{T}_{t\rightarrow\tau}
=
\mathbf{T}_{t}^{-1}\mathbf{T}_{\tau}.
\]
The scalar $w_{\tau}$ is an absolute normalized gripper command rather than a relative gripper displacement. For the teleoperation data used in this work, $w_{\tau}\in[0,1]$.
```

### 修改原因

- 实际 action 为 10 维：3D 平移、6D 旋转和 1D 夹爪。
- 相对位姿由 $\mathbf{T}_{t}^{-1}\mathbf{T}_{\tau}$ 定义，因此是在当前工具坐标系中表达。
- 夹爪维度保持绝对量，不随 TCP 一起转换为相对动作。
- 对当前 `teleop_raw` 数据，$[0,1]$ 表示归一化夹爪命令，而不是以米或毫米表示的物理宽度。

---

## 10. Relative coordinates and generalization

### 原句

```latex
Formulating trajectories in the local relative coordinate frame improves cross-environment spatial generalization.
```

### 修改后

```latex
Representing trajectories in the local tool-relative coordinate frame reduces their dependence on absolute robot-base coordinates and is intended to improve spatial generalization across different scene placements.
```

### 修改原因

- “improves”是确定性的实验结论，需要专门的跨环境对比实验支持。
- “is intended to improve”更适合作为方法动机。

如果论文已有充分的跨环境实验，可以保留更强的结论，但应在句末引用对应实验章节或表格。

---

## 11. Receding-horizon execution

### 原句

```latex
The policy is executed in a receding-horizon control loop. Although the diffusion model generates an entire chunk of $n$ future steps, the robot executes only the immediate $k$ steps ($k \le n$) before re-sampling observations and re-conditioning the policy.
```

### 修改后

```latex
Policy inference and robot control run asynchronously in a receding-horizon loop. At each inference cycle, the latest observation history is used to predict a future action chunk. Predicted chunks are periodically inserted into a temporal-ensemble buffer, from which the controller selects one ensembled target at each control step.
```

### 修改原因

- 当前实现不是简单执行预测序列的前 $k$ 步并丢弃其余动作。
- 每次预测得到一个未来 action chunk。
- 新旧预测在 temporal-ensemble buffer 中进行融合。
- 控制线程每个控制周期从 buffer 中取出一个融合后的目标。

---

## 12. Closed-loop response

### 原句

```latex
This continuous closed-loop replenishment allows the policy to react dynamically to instantaneous contact events---such as jamming during tight pin-slot insertion or slippage during grasping---at high control frequencies.
```

### 修改后

```latex
Repeated inference with updated tactile observations allows subsequent action targets to be corrected in response to contact events, such as jamming during tight pin-slot insertion or slippage during grasping.
```

### 修改原因

- “instantaneous”过强，因为触觉反馈需要等待下一次推理和动作更新。
- 控制频率与模型推理频率不同，不能笼统地声称模型以高控制频率即时重新规划。

如果需要报告当前实验频率，可以增加：

```latex
In our implementation, robot targets are executed at 12 Hz, while policy inference is performed at 6 Hz.
```

---

## 13. Observation Modality Variants：设置名称

### 原句

```latex
\textbf{Observation Modality Variants:} In addition to the standard closed-loop setting where $\mathbf{O}*t = {o*{t-1}, o_t}$ updates continuously, we investigate a \textit{vision-deprived tactile control} setting in the tactile only wiping task in Section~\ref{sec:contact_rich_manipulation}.
```

### 修改后

```latex
\textbf{Observation Modality Variants:}
In addition to the standard closed-loop setting, in which
$\mathbf{O}_t=\{o_{t-1},o_t\}$ is updated continuously, we investigate a
\textit{first-frame-vision tactile control} setting for the tactile-only wiping task in Section~\ref{sec:contact_rich_manipulation}.
```

### 修改原因

- 该设置仍然使用初始视觉帧，因此不是完全的 vision-deprived control。
- `first-frame-vision` 或 `initial-frame-vision` 与实现更加一致。
- 修正 observation history 的公式格式。

---

## 14. First-frame variant

### 原句

```latex
Under this scenario, visual input is restricted strictly to the initial scene frame ($\mathbf{O}_t = {o_0}$), forcing the diffusion policy to guide fine-grained action entirely through closed-loop proprioceptive and \modelname tactile dynamics.
```

### 修改后

```latex
In this setting, the visual feature extracted from the initial scene frame $o_0$ is cached and reused throughout the episode, while the temporal \modelname tactile observations continue to be updated. Under the current implementation, the TCP pose and gripper state are provided only as initial context and are not continuously updated as neural-policy inputs. The latest measured TCP pose is nevertheless used by the execution layer to transform relative policy actions into absolute robot targets.
```

### 修改原因

- 推理时缓存并重复使用的是初始图像的视觉特征。
- 当前 first-frame 配置只有触觉低维输入保持时间更新。
- TCP 位姿和夹爪状态不作为持续变化的神经网络条件，因此不能写成 closed-loop proprioceptive dynamics。
- 执行层仍使用最新 TCP 测量值，将相对动作转换成绝对机器人目标。

如果论文确实希望声称策略持续使用 closed-loop proprioception，则需要修改 first-frame encoder 的 `temporal_low_dim_keys`，将 TCP 和夹爪状态加入持续输入，并重新训练相应模型。

---

# 可直接替换的完整 LaTeX 版本

下面版本按照当前标准策略和 first-frame 策略实现统一整理。触觉部分默认使用双磁传感器表述；如果实验使用单传感器，可以删除与传感器 2 相关的内容。

```latex
\subsection{Multimodal Policy Architecture}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/exp_figures/network1.png}
\caption{\textbf{Pipeline of the multimodal conditional diffusion policy.}
Wrist-camera images are encoded by a ResNet-18 backbone with global average pooling, while normalized low-dimensional tactile and proprioceptive vectors are concatenated directly without dedicated encoders. Features from the observation history are flattened into a global conditioning vector, which modulates a 1D convolutional U-Net through feature-wise linear modulation (FiLM) to denoise future action trajectories.}
\label{fig:policy_architecture}
\vspace{-10pt}
\end{figure}

The policy employs a conditional 1D convolutional U-Net that denoises an action sequence conditioned on a fixed-length multimodal observation history (Fig.~\ref{fig:policy_architecture}). At each control timestep $t$, the multimodal observation comprises the following components:

\begin{itemize}
    \item \textit{Visual Observations}: The standard closed-loop policy receives an observation history
    $\mathbf{O}_t=\{o_{t-1},o_t\}$ containing the two most recent RGB frames captured by the wrist-mounted camera. Each frame is processed independently by a ResNet-18 backbone~\cite{he2016deep}, whose global average pooling layer produces a 512-dimensional visual feature.

    \item \textit{Tactile Observations}: The policy receives temporal measurements from two magnetic tactile sensors. For sensor $j\in\{1,2\}$, the tactile history is
    $\mathbf{B}^{(j)}_t=\{\mathbf{b}^{(j)}_{t-1},\mathbf{b}^{(j)}_t\}$.
    Each $\mathbf{b}^{(j)}_{\tau}$ is obtained by temporally averaging the baseline-subtracted three-axis readings from its magnetometer channels and flattening them into a 15-dimensional tactile vector. The two sensor vectors are independently standardized using statistics computed from the training set and concatenated as
    $\mathbf{b}_{\tau}=[\mathbf{b}^{(1)}_{\tau},\mathbf{b}^{(2)}_{\tau}]$.
    Unlike vision-based tactile approaches that employ pretrained tactile CNNs or vision transformers, these low-dimensional tactile vectors are concatenated directly with the other observation features without a dedicated tactile encoder.

    \item \textit{Proprioceptive State}: The standard policy receives a proprioceptive history
    $\mathbf{P}_t=\{\mathbf{p}_{t-1},\mathbf{p}_t\}$. Each state $\mathbf{p}_{\tau}$ contains the end-effector position, a continuous 6D rotation representation, and the continuous gripper state. When relative-action training is enabled, the end-effector poses in the observation window are expressed relative to the most recent observed pose.
\end{itemize}

For each observation step $\tau$, the visual, tactile, and proprioceptive features are concatenated as
\begin{equation}
\mathbf{z}_{\tau}
=
[\operatorname{ResNet18}(o_{\tau}),
 \tilde{\mathbf{b}}_{\tau},
 \tilde{\mathbf{p}}_{\tau}],
\end{equation}
where the tilde denotes normalization. The features from the two observation steps are flattened into a single global conditioning vector,
\begin{equation}
\mathbf{c}_t
=
\operatorname{Flatten}
(\mathbf{z}_{t-1},\mathbf{z}_t).
\end{equation}
The conditioning vector is combined with the diffusion-timestep embedding and modulates the residual blocks of the 1D U-Net through feature-wise linear modulation (FiLM)~\cite{perez2018film}. The architecture does not employ cross-attention.


\subsection{Action Representation and Closed-Loop Execution}

Following the relative-action formulation used in UMI~\cite{chi2024universal}, the policy predicts a sequence of $n$ future end-effector waypoints,
\begin{equation}
\mathbf{A}_t
=
\{a_{t+1},\ldots,a_{t+n}\}.
\end{equation}
Each action waypoint is represented as
\begin{equation}
a_{\tau}
=
[\Delta\mathbf{p}_{\tau},
 \mathbf{r}^{6D}_{\tau},
 w_{\tau}]
\in\mathbb{R}^{10},
\end{equation}
where $\Delta\mathbf{p}_{\tau}\in\mathbb{R}^{3}$ and
$\mathbf{r}^{6D}_{\tau}\in\mathbb{R}^{6}$ represent the relative end-effector transform
\begin{equation}
\Delta\mathbf{T}_{t\rightarrow\tau}
=
\mathbf{T}_{t}^{-1}\mathbf{T}_{\tau}.
\end{equation}
The scalar $w_{\tau}$ is an absolute normalized gripper command rather than a relative gripper displacement. For the teleoperation data used in this work, $w_{\tau}\in[0,1]$. Representing trajectories in the local tool-relative coordinate frame reduces their dependence on absolute robot-base coordinates and is intended to improve spatial generalization across different scene placements.

Policy inference and robot control run asynchronously in a receding-horizon loop. At each inference cycle, the latest observation history is used to predict a future action chunk. Predicted chunks are periodically inserted into a temporal-ensemble buffer, from which the controller selects one ensembled target at each control step. Repeated inference with updated tactile observations allows subsequent action targets to be corrected in response to contact events, such as jamming during tight pin-slot insertion or slippage during grasping. In our implementation, robot targets are executed at 12 Hz, while policy inference is performed at 6 Hz.

\textbf{Observation Modality Variants:}
In addition to the standard closed-loop setting, in which
$\mathbf{O}_t=\{o_{t-1},o_t\}$ is updated continuously, we investigate a
\textit{first-frame-vision tactile control} setting for the tactile-only wiping task in Section~\ref{sec:contact_rich_manipulation}. In this setting, the visual feature extracted from the initial scene frame $o_0$ is cached and reused throughout the episode, while the temporal \modelname tactile observations continue to be updated. Under the current implementation, the TCP pose and gripper state are provided only as initial context and are not continuously updated as neural-policy inputs. The latest measured TCP pose is nevertheless used by the execution layer to transform relative policy actions into absolute robot targets.
```

## 实现依据

上述修改主要对应以下实现：

- `reactive_diffusion_policy/model/vision/model_getter.py`：ResNet-18 保留全局平均池化，仅移除最终 FC。
- `reactive_diffusion_policy/model/vision/multi_image_obs_encoder.py`：视觉、触觉和本体特征直接拼接。
- `reactive_diffusion_policy/model/vision/first_frame_obs_encoder.py`：初始视觉缓存以及 tactile-only temporal low-dimensional input。
- `reactive_diffusion_policy/policy/diffusion_unet_image_policy.py`：观察历史特征展平为 global condition。
- `reactive_diffusion_policy/model/diffusion/conditional_unet1d.py`：FiLM 条件残差块，没有 cross-attention。
- `reactive_diffusion_policy/common/action_utils.py`：相对位姿的坐标变换定义。
- `reactive_diffusion_policy/env_runner/real_runner.py`：异步推理、relative-to-absolute 转换和 temporal ensemble 执行。
