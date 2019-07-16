---
title: DeRPN
date: 2019-07-15 15:04:18
tags: object detection
---
论文 [DeRPN: Taking a further step toward more general object detection](https://arxiv.org/abs/1811.06700)

two-stage SOTA 目标检测器通常会使用 anchor，比如 Faster R-CNN 中的 RPN，但是对于不同的数据集，则需要重新设计超参数，如 anchor 的 scale 和 aspect ratio，并且一旦选定就固定了，这在被检测目标尺度变化较大时，检测性能往往不理想，当然，也有人尝试使用 K-means 聚类计算得到 anchor，但是对最终的检测性能的提升非常有限。本文提出 DeRPN 用于解决 RPN 的这一不足之处，如图 1(b)，
![](/images/DeRPN_fig1.png)

DeRPN 通过分离宽度和高度来分解检测维度（维度分解）。利用灵活的 anchor strings（不理解这个概念没关系，阅读完下一节就理解了），使得可以选择最佳 anchor 来匹配目标。

# 方法论
## 建模
我们知道目标检测网络通常都是一个 CNN 网络用于抽取特征，记抽取到的特征为 $\mathbf x$，然后经过两个并行的检测分支：回归和分类，其中回归是在 anchor box （$B_a$）基础上进行回归得到目标位置，而分类分支则在最后的预测值上应用 sigmoid（二分类）或 softmax（多分类），记此函数为 $\sigma$，从而得到 bbox 的分类置信度（概率），用数学语言描述则为：
$$\mathbf {t = W_t x+b_r}
\\\\ B(x,y,w,h)=\psi(\mathbf t, B_a(x_a,y_a,w_a,h_a))
\\\\ P_B=\sigma(\mathbf {W_c x + b_c})$$
其中 $\mathbf {W_r, b_r}$ 表示回归分支的权重和偏置，$\mathbf {W_c, b_c}$ 表示分类分支的权重和偏置，$\psi$ 表示预测 box 的位置解码，例如 Faster R-CNN 中根据位置偏差 $\mathbf t$ 和 region proposals 的坐标计算出预测 box 的坐标。

显然由于目标形状的多样性，anchor 的数量会非常大，这不利于训练，而且我们也很难设计出合适的 anchor 形状，所以当 anchor 严重偏离 gt box 时，检测性能下降! 目标检测的维度分解具体是指分离宽度和高度，以减轻目标不同尺度带来的影响。我们引入 anchor string，$(S_a^w(x_a,w_a), S_a^h(y_a,h_a))$，各自分别作为目标宽度和高度的回归参照，anchor string 分别独立预测 $(S_w(x,w), S_h(y,h))$ 以及对应的分类概率 $(P_s^w, P_s^h)$，此过程的数学语言描述为，
$$\mathbf t^w=\mathbf {W_r}^w \mathbf {x+ b_r}^w \qquad S_w(x,w)=\psi(\mathbf t^w, S_a^w(x_a,w_a))
\\\\ \mathbf t^h=\mathbf {W_r}^h \mathbf {x+ b_r}^h \qquad S_h(x,w)=\psi(\mathbf t^h, S_a^h(y_a,h_a))
\\\\ P_s^w=\sigma (\mathbf {W_c}^w \mathbf {x+b_c}^w) \qquad P_s^h=\sigma (\mathbf {W_c}^h \mathbf {x+b_c}^h)$$
相比上一组计算式，容易看出确实是将宽度和高度分类开来（包括分类概率也分解为两个维度上各自独立的分类概率）。现在我们从分解开来的两个维度预测恢复出 bbox 的位置以及分类置信度，
$$B(x,y,w,h)=f(S_w(x,w),S_h(y,h))
\\\\ P_B=g(P_s^w, P_s^h)$$
其中，f 表示合并两个维度的一种策略函数，g 计算合并后 bbox 的分类置信度（可以是算术平均，或调和平均）。
### 匹配复杂度
假设数据集中目标的宽度或高度共有 n 种情况，那么一共有 $n^2$ 种情况需要 anchor box 去匹配，即，匹配复杂度为 $O(n^2)$，而在维度分解下，n 种宽度和高度分别独立地由 anchor string 去匹配，匹配复杂度降为 $O(n)$。

## 维度分解
### Anchor strings
RPN 以 anchor string 作为回归参照，DeRPN 则将二维 box 拆分为两个独立的一维部分作为回归参照，称为 anchor string。虽说 anchor string 可以匹配任意object 的宽度或高度，设置 anchor string 为一个等比数列 $\{a_n\}$，例如 (16,32,64,128,256,512,1024)，此时可用于匹配目标宽度或高度的范围为 $[8\sqrt 2,1024 \sqrt 2]$，通常这已经足够覆盖很多场景下的目标尺寸了。解释一下这个的 $\sqrt 2$，记一个 anchor string 长度值（等比数列中的一项）为 $a_i$，这个 anchor string 可匹配的目标边长范围为$[a_i/\sqrt 2, a_i\sqrt 2]$，由于等比数列中公比为2，此时这个等比数列中各项所匹配的目标边长范围无缝连接，形成一个大的范围 $[8\sqrt 2,1024 \sqrt 2]$。

图 2 为 DeRPN 网络，
![](/images/DeRPN_fig2.png) <center>(a) 目标宽度和高度分别独立使用 anchor string 匹配，粗线表示匹配较好的 anchor string；(b) 在 anchor string 上应用分类和回归，虚线表示置信度低的 anchor string；(c) 合并预测的宽度和高度生成 bbox；(d) 使用置信度阈值和 NMS 过滤得到 region proposals。</center>

如何为目标选择最佳匹配的 anchor string？在 RPN 中，通过 anchor box 与 gt box 的 IoU 决定是否选择 anchor 参与训练。比如， anchor 的最大 IoU 超过 0.7，或者 gt 的最大 IoU 对应的 anchor 均可作为正例。在 DeRPN 中则基于长度将 anchor string 与目标进行匹配，评估最佳匹配 anchor string 的方法为，
$$M_j=\{i|\arg \min_i |\log e_j - \log a_i|\} \cup \{i,i+1| \begin{vmatrix}\frac {e_j} {a_i} - \sqrt q \end{vmatrix} \le \beta\}, \ (i=1,...,N) \quad(9)$$
$M_j$ 表示与第 j 个目标匹配的 anchor string 的索引，$e_j$ 是目标边长（宽或高），N 是等比数列 $\{a_n\}$ 中的项数，q 是等比数列的公比（本文中设置为 2）。

上式中，第一项表示选择与目标边长最接近的 anchor string，这是一种很直观的选择策略，然而还有第二种选择策略，见上式第二项，我们将条件约束稍作变形得 $(\sqrt q-\beta)\times a_i \le e_j \le (\sqrt q+\beta)\times a_i$，范围 $[(\sqrt q-\beta)\times a_i, (\sqrt q+\beta)\times a_i]$ 称为 i 关联的转移区间，$\beta$ 控制区间长度，如果目标边长 $e_j$ 位于此范围内，那么选择 i 和 i+1 作为匹配的 anchor string 的索引。

上文我们说到 $a_i$ 可匹配的目标边长范围为 $[a_i/ \sqrt q,a_i\sqrt q]$，按道理说，如果 $e_j$ 落于这个区间，就选择 i 作为匹配的索引就好了鸭（不考虑边长等于区间端点值的情况，事实上这种情况的可能性为0），但是考虑到图像噪声和 gt 标记偏离正确位置等因素，按照这个选择策略选择的 i 不一定准确，而图像噪声和 gt 标记偏离正确位置等因素所带来的影响相对较小，所以我们选择连续的两个 anchor string 索引即可保证目标能落入这两个连续 anchor string 的可匹配范围，$a_i, a_{i+1}$ 的可匹配范围为 $[a_i/\sqrt q, a_i \sqrt q] \cup [a_i \sqrt q,qa_i\sqrt q]$，其（非几何）“中心”为 $a_i \sqrt q$，所以很自然地，如果目标边长 $e_j$ 在这个“中心”附近，就选择 i 和 i+1 作为匹配索引，判断是否在附近的条件不难理解，
$$(\sqrt q-\beta)\times a_i \le e_j \le (\sqrt q+\beta)\times a_i$$
剩下的就不多说了。

忽略转移区间，可以知道 anchor string 与目标边长之间的最大偏移比例为 $\sqrt q$（如果考虑转移区间，最大偏移比例则为 $\max(\sqrt q + \beta, q/(\sqrt q-\beta))$，也就比 $\sqrt q$ 大一点点），这表示 DeRPN 中回归损失是有界的，而 RPN 中较小的 IoU 则会导致较大的回归损失，经验表明，如果 anchor box 严重偏离 gt，RPN 甚至无法收敛

### Label assignment
对齐的 anchor string 位于 feature map 上目标中心处，其中与目标匹配较好的（根据式 (9)）则标记为正。除了对齐的 anchor string，还使用了 observe-to-distribute 策略来选择其他 anchor string：1. 观察每个 anchor string 的回归结果，回归之后，结合宽度/高度的预测得到 region proposal，如果这个 region proposal 与某个 gt 的 IoU 大于一定阈值（0.6），那么就将正标签分发到对应的 anchor string 上。不满足以上任何条件的 anchor string 则标记为负。

### Consistent network
DeRPN 与 RPN 的网络结构是一致的，故可方便地移植到当前 two-stage 目标检测器中。如图 2 所示，由一个 3x3 的卷积层，后跟两个并列的 1x1 卷积层，分别用于分类和回归，组成了 DeRPN 网络。记 anchor string 长度的等比数列为 $\{a_n\}$，数量为 N，宽度和高度独立使用 anchor string，分类预测 $2\times 2N$ 个得分来估计 anchor string 是否匹配目标边长（二值分类置信度），anchor string 预测目标的宽需要两个值 $(x,w)$，同理对于目标的高也需要两个值 $(y,h)$，故回归一共预测 $2 \times 2N$ 个值。

### Scale-sensitive loss function
目标的尺度分布不是均匀的，大目标比小目标更多。如果简单地将目标混合起来计算损失，那么小目标对损失的影响将会被大目标带来的影响所淹没，本文提出一种新型的尺度敏感的损失函数，公平地对待不同尺度的目标，
$$L(\{p_i\},\{t_i\})=\sum_{j=1}^N \sum_{i=1}^M \frac 1 {|R_j|} L_{cls}(p_i,p_i^*) \cdot \Bbb I\{i \in R_j\} + \lambda \sum_{j=1}^N \sum_{i=1}^M \frac 1 {|G_j|} L_{reg} (t_i,t_i^*)\cdot \Bbb I\{i \in G_j\} \quad (10)
\\\\ R_j=\{k|s_k=a_j, k=1,...,M\} \quad (11)
\\\\ G_j=\{k|s_k \in A, s_k=a_j, p_i^*=1, k=1,...,M\} \quad (12)$$

这里，N 是等比数列的项数，M 是 batch size，s 表示 anchor string，$p_i$ 表示一个批次中第 i 个 anchor string 的预测概率，$p_i^*$ 表示 gt label，当 anchor string 为正时等于 1， 否则等于 0。$t_i$ 表示参数化坐标的预测向量，$t_i^*$ 为相应的 gt 向量。A 表示对齐的 anchor string 集合。$R_j$ 这个索引集包含了具有相同尺度的 anchor string，其中 j 用于指示尺度 $a_j$。$G_j$ 这个索引集包含了具有相同尺度的对齐的正 anchor string，同样 j 用于指示尺度 $a_j$。上式表明每个尺度下的目标损失均根据这个尺度下的 anchor string 数量进行归一化，这可以有效地避免小目标优化作用被大目标淹没。分类损失使用交叉熵，回归损失使用 smooth L1 损失，
$$L_{cls}(p_i,p_i^*)=- p_i^*\log p_i-(1-p_i^*)\log (1-p_i)
\\\\ L_{reg}(t_i,t_i^*)=\sum_{j \in \{x,y,w,h\}} smooth_{L_1}(t_i^j,t_i^{j*})$$

预测值 t 表示坐标偏差，这一点与 Fast/Faster R-CNN 中完全一样，故可根据下式解码出预测 box 坐标，
$$x=x_a+w_a \times t_x \quad (13)
\\\\ y=y_a+h_a \times t_y \quad (14)
\\\\ w=w_a \times e^{t_w} \qquad (15)
\\\\ h=h_a \times e^{t_h} \qquad (16)$$

# 维度合并
DeRPN 使用维度分解来预测，然而最终的 region proposal 是二维的 bbox，故需要合并宽和高以恢复出 region proposal。

__像素级别的合并算法__ 根据预测坐标偏差 t 和 anchor string 可以解码出宽和高，记所有预测宽的集合为 W，根据预测宽的概率选择 top-N，记 $W_N$，对于这 top-N 中任意一个宽的预测 (x,w)（对应的概率为 $p^W$），我们在 (x,w) 所在的像素位置处选择 top-k 的目标高的预测 $(y^{(k)},h^{(k)})$，于是得到一系列的 bbox $B_w=\{(x,y^{(k)},w,h^{(k)}\}$，每个组合后的 bbox 的概率使用调和平均计算得到，
$$p^B=2/ \left(\frac 1 {p^W}+\frac 1 {p^H}\right)$$
其中 $p^W$ 为 (x,w) 对应的预测概率，$p^H$ 为 $(y^{(k)},h^{(k)})$ 对应的预测概率。

类似地，对于 top-N 预测概率的目标高 $H_N$，按上面的策略选择得到 $B_h=\{(x^{(k)},y,w^{(k)},h\}$，对这两个集合的并 $B=B_w \cup B_h$ 使用 NMS，然后再选择 top-M 作为 region proposals。尽管这个合并过程引入了一些背景 bbox，但是第二 stage 的目标检测器可以通过分类分支抑制它们。

# 实验
请阅读原文，略。

# 结论
1. 介绍了 DeRPN，将目标的宽和高两个维度进行分解
2. 使用了新型损失函数，避免了小目标（少数）的优化作用被大目标（多数）淹没