# T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models

## Background
**Task**: Aim to “dig out” the capabilities that T2I models have implicitly learned, especially the high-level structure and semantic capabilities, and then explicitly use them to control the generation more accurately.

<center>

![](./imgs/T2I-Adapter_Fig1.png)
</center>

Figure 1. Thanks to the T2I-Adapter, we can generate more imaginative results that the original T2I model (e.g., Stable Diffusion) can hardly generate accurately, e.g., $\textit{"A car with flying wings"}$ and $\textit{"Iron Man with bunny ears"}$. Various guidance such as color, depth, sketch, semantic segmentation, and keypose can be used. We can further achieve local editing and composable guidance with our T2I-Adapter.

**Idea**: 
- A small adapter model can achieve this purpose, as it is not learning new generation abilities, but learning a mapping from control information to the internal knowledge in T2I models.
- Propose the T2I-Adapter, which is a lightweight model and can be used to learn this alignment with a relatively small amount of data.
- In this way, we can train various adapters according to different conditions and they can provide more accurate and controllable xtra generation guidance for the pre-trained T2I diffusion models.

<center>

![](./imgs/T2I-Adapter_Fig2.png)
</center>

Figure 2. The T2I-Adapters, as extra networks to inject guidance information while not affecting their original generation ability.

**The properties of the T2I-Adapters**
- $\textbf{\textit{Plug-and-play.}}$ Not affect original network topology and generation ability
- $\textbf{\textit{Simple and small.}}$ ∼ 77 M parameters and ∼ 300 M storage 
- $\textbf{\textit{Flexible.}}$ Various adapters for different control conditions
- $\textbf{\textit{Composable.}}$ Several adapters to achieve multi-condition control
- $\textbf{\textit{Generalizable.}}$ Can be directly used on customed models

## Contributions
- We propose T2I-Adapter, a simple, efﬁcient yet effective method to well align the internal knowledge of T2I models and external control signals with a low cost.
- T2I-Adapter can provide more accurate controllable guidance to existing T2I models while not affecting their original generation ability. 
- Extensive experiments demonstrate that our method works well with various conditions and these conditions can also be easily composed to achieve multicondition control. 
- The proposed T2I-Adapter also has an attractive generalization ability to work on some custom models and coarse conditions e.g., free-hand style sketch.

## Related Work
- There are no papers about autonomous driving

## Method
### Preliminary: Stable Diffusion
- SD (Stable Diffusion) is a two-stage diffusion model, which contains an autoencoder and a U-Net denoiser. The optimization process can be defined as the following formulation:
$$\mathcal{L}=\mathbb{E}_{\mathbf{Z}_t,\mathbf{C},\epsilon,t}(||\epsilon-\epsilon_\theta(\mathbf{Z}_t,\mathbf{C})||_2^2),\tag{1}$$

  - $\mathbf{C:}$ the conditional information
  - $\mathbf{Z}_t=\sqrt{\overline{\alpha}_t}\mathbf{Z}_0+\sqrt{1-\overline{\alpha_t}}\epsilon, \epsilon\in\mathcal{N}(0,\mathbf{I}):$ the noised feature map at step t; During inference, the input latent map $Z_{t}$ is generated from random Gaussian distribution 
  - $\epsilon_{\theta}:$ the function of UNet denoiser; Given $Z_{t}$, $\epsilon_{\theta}$ predicts a noise estimation at each step t, conditioned on $\mathbf{C}$
- In the conditional part, SD utilized the pre-trained CLIP text encoder to embed text inputs to a sequence of token $\boldsymbol{y}$. Then it utilizes the cross-attention model to combine $\boldsymbol{y}$ into the denoising process. It can be deﬁned as the following formulation:
$$\left\{\begin{array}{l}\mathbf{Q}=\mathbf{W}_Q\phi(\mathbf{Z}_t); \mathbf{K}=\mathbf{W}_K\tau(\mathbf{y}); \mathbf{V}=\mathbf{W}_V\tau(\mathbf{y})\\Attention(\mathbf{Q},\mathbf{K},\mathbf{V})=softmax(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}})\cdot\mathbf{V},\end{array}\right.\tag{2}$$

  - $\phi(\cdot)$ and $\tau(\cdot)$ are two learnable embeddings
  - $\mathbf{W}_{Q},$ $\mathbf{W}_{K},$ and $\mathbf{W}_{V}$ are learnable projection matrices

### Overview of T2I-Adapter
- An overview of our method is presented in Fig. 3, which is composed of a pre-trained SD model and several T2I adapters. The adapters are used to extract guidance features from different types of conditions. The pre-trained SD has ﬁxed parameters to generate images based on the input text feature and extra guidance feature.

![](./imgs/T2I-Adapter_Fig3.png)
Figure 3. The overall architecture is composed of two parts: (1) a pre-trained stable diffusion model with ﬁxed parameters; (2) several T2I-Adapters trained to align internal knowledge in T2I models and external control signals. Different adapters can be composed by directly adding with adjustable weight $\omega$. 

- The detailed architecture of T2I-Adapter is shown in the lower right corner. 
  - $\textit{\textbf{Pixel Unshuffle}}:$ The original condition input has the resolution of 512 × 512. Here, we utilize the pixel unshufﬂe operation to downsample it to 64 × 64. 
  - $\textit{\textbf{One Convolutional Layers (Conv) and Two Residual blocks (RB)}}:$ In each scale, one convolution layer and two residual blocks (RB) are utilized to extract the condition feature $\mathbf{F}_c^k$.  
  - $\textit{\textbf{Downsample}}:$ Reduces the resolution of the features at different scales (from 64x64 down to 8x8). Each scale provides different levels of detail, which helps the model capture both fine and coarse features of the condition input.

<center>

![](./imgs/T2I-Adapter_Fig4.png)
</center>

Figure 4. In complex scenarios, SD fails to generate accurate results conforming to the text. In contrast, our T2I-Adapter can provide structure guidance to SD and generate plausible results.

### Adapter Design
- Our proposed T2I-adapter is composed of four feature extraction blocks and three downsample blocks to change the feature resolution. 
- Multi-scale condition features $\mathbf{F}_{c} = \{\mathbf{F}_{c}^{1},\mathbf{F}_{c}^{2},\mathbf{F}_{c}^{3},\mathbf{F}_{c}^{4}\}$ are formed. 
- The intermediate feature in the encoder of UNet denoiser is $\mathbf{F}_{enc}= \{\mathbf{F}_{enc}^{1},\mathbf{F}_{enc}^{2},\mathbf{F}_{enc}^{3},\mathbf{F}_{enc}^{4}\}$
- The dimension of $\mathbf{F}_{c}$ is same as $\mathbf{F}_{enc}$.
- $\mathbf{F}_c$ is then $\textbf{added}$ with $\mathbf{F}_{enc}$ at each scale.

$$\mathbf{F}_c=\mathcal{F}_{AD}(\mathbf{C})\tag{3}$$

$$\hat{\mathbf{F}}_{enc}^i=\mathbf{F}_{enc}^i+\mathbf{F}_c^i, i\in\{1,2,3,4\}\tag{4}$$

<center>

$\textbf{C}:$ the condition input
$\mathcal{F}_{AD}:$ the T2I-Adapter
</center>

**Multi-adapter controlling**
$$\mathbf{F}_c=\sum_{k=1}^K\omega_k\mathcal{F}_{AD}^k(\mathbf{C}_k),\tag{5}$$

<center>

$k\in[1,K]$ represents the $k$-th guidance
$\omega_k$ is the adjustable weight to control the composed strength of each adapter
</center>

### Model Optimization
- During optimization, we fix the parameters in SD and only optimize the T2I-adapter.
- Each training sample includes the original image $\textbf{X}_0$, condition map $\textbf{C}$ and text prompt $y$.
- The optimization process: given an image $\textbf{X}_0$, we ﬁrst embed it to the latent space $\textbf{Z}_0$ via the encoder of autoencoder. Then, we randomly sample a time step $t$ from $[0, T]$ and add corresponding noise to $\textbf{Z}_0$, producing $\textbf{Z}_t$. 
$$\mathcal{L}_{AD}=\mathbb{E}_{\mathbf{Z}_{0},t,\mathbf{F}_{c},\epsilon\sim\mathcal{N}(0,1)}\left[||\epsilon-\epsilon_{\theta}(\mathbf{Z}_{t},t,\tau(\mathbf{y}),\mathbf{F}_{c})||_{2}^{2}\right]\tag{6}$$

**Non-uniform time step sampling during training.**
- Introducing time embedding into the adapter is helpful for enhancing the guidance ability. 

<center>

![](./imgs/T2I-Adapter_Fig5.png)
</center>

Figure 5. We evenly divide the DDIM inference sampling into 3 stages, i.e., beginning, middle and late stages. We then add guidance information to each of the three stages. We ﬁnd that adding guidance in the middle and late stages had little effect on the result. 

- A conclusion can be found from Figure. 5: The main content of the generation results is determined in the early sampling stage and the guidance information will be ignored during training if $t$ is sampled from the later section.  
- To strengthen the training of adapter, non-uniform sampling is adopted to increase the probability of $t$ falling in the early stage.
$$t = (1 - (\frac{t}{T})^{3}) \times T, t \in U(0,T)\tag{7}$$

<center>

the cubic function as the distribution of $t$: Cubic sampling is a method that transforms a uniformly distributed $t$ through a cubic function to generate a distribution that favors the early stages.
</center>

- The comparison between uniform sampling and cubic sampling is presented in Fig. 6.

<center>

![](./imgs/T2I-Adapter_Fig6.png)
</center>

Figure 6. The effect of cubic sampling during training. The uniform sampling of time steps has the problem of weak guidance, especially in color controlling. The cubic sampling strategy can rectify this weakness.

### Limitation
- In the case of multi-adapter control, the combination of guidance features requires manual adjustment.