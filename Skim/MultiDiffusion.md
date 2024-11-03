# MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation

## Background
**Task**: Using a framework called MultiDiffusion to achieve controllable and diverse image generation without further training or fine-tuning. 

**Previous Approach to control specificed image generation**
- training a model from scratch or finetuning a given diffusion model for the task at hand (e.g., inpainting, layout-to-image training, etc. )
-  Reuse a pre-trained model and add some controlled generation capability.

**The current basic idea of MultiDiffusion**
MultiDiffusion defines a new generation process, which consists of multiple reference diffusion generation processes applied to different regions of the generated image. Each region undergoes a denoising sampling step, and then MultiDiffusion performs a global denoising sampling step, integrating the denoising directions of each region through a "least squares optimal solution." This process finds an optimal solution that minimizes the differences between regions and considers the common pixels shared between adjacent regions, achieving a seamless transition and ultimately producing a unified panoramic image. With MultiDiffusion, a pre-trained text-to-image diffusion model can be leveraged for various tasks, including synthesizing images at the desired resolution or aspect ratio, or generating images based on rough region-based text prompts, as shown in Fig. 1.

![](./imgs/MultiDiffusion_Fig1.png)
Figure 1.MultiDiffusion enables ﬂexible text-to-image generation, unifying multiple controls over the generated content, including desired aspect ratio, or simple spatial guiding signals such as rough region-based text-prompts.

## Related Work
- There are no papers about autonomous driving

## Method
### A pre-trained diffusion model serving as a reference model

$$\Phi:\mathcal{I}\times\mathcal{Y}\to\mathcal{I}$$

- $\mathcal{I}$: image space, $\mathcal{I}=\mathbb{R}^{H\times W\times C}$
- $\mathcal{Y}$: condition space, e.g., $y\in\mathcal{Y}$ is a text prompt
- Initializing $I_{T}\sim P_{\mathcal{I}}: P_{\mathcal{I}}$ represents the distribution of Gaussian i.i.d. pixel values

$$I_T,I_{T-1},\ldots,I_0\quad\mathrm{s.t.}\quad I_{t-1}=\Phi(I_t|y)\tag{1}$$

- gradually transforming the noisy image $I_{T}$ into a clean image $I_{0}$

### MultiDiffusion

**Goal**: 
- Leverage $\Phi$ to generate images in a potentially different image space $\mathcal{J}=\mathbb{R}^{H^{\prime}\times W^{\prime}\times C}$ and condition space $\mathcal{Z}$, without any training or finetuning

**Definition**:
- MultiDiffuser: 
$$\Psi:\mathcal{J}\times\mathcal{Z}\to\mathcal{J}$$
- The MultiDiffusion process: starts with some initial noisy input $J_T\sim P_{\mathcal{J}}$, where $P_\mathcal{J}$ is a noise distribution over $\mathcal{J}$, and produces a series of images $$J_T,J_{T-1},\ldots,J_0\quad\mathrm{s.t.}\quad J_{t-1}=\Psi(J_t|z)\tag{2}$$

**Key Idea**：
- $F_{i}:\mathcal{J}\to\mathcal{I}$ where $i\in[n]=\{1,\ldots,n\}$: define a set of mappings between the target and reference image spaces
- $\lambda_{i}:\mathcal{Z}\to\mathcal{Y}$ where $i\in[n]=\{1,\ldots,n\}$: a corresponding set of mappings between the condition spaces
- The key idea is to deﬁne $\psi$ to make every MultiDiffuser step $J_{t-1}=\Psi(J_{t}|z)$ follow as closely as possible $\Phi(I_t^i|y_i)$, $i\in[n]$.
  $i$: the image is separated into $i$ parts
  i.e.,the denoising steps of $\phi$ when applied to the images and conditions:

$$I_t^i=F_i(J_t),\quad y_i=\lambda_i(z)$$

- The process is given by solving the following optimization problem:

$$\Psi(J_t|z)=\underset{J\in\mathcal{J}}{\arg\min} \mathcal{L}_{\mathrm{FID}}(J|J_t,z)\tag{3}$$

$\hspace*{2em}$ The minimum $J$ represents the optimal representation of the entire image at this time step $t$, $\hspace*{2em}$ ensuring that all regions are coordinated during the denoising process, thereby producing a $\hspace*{2em}$  seamlessly integrated final image.

$$\mathcal{L}_{\mathrm{FTD}}(J|J_t,z)=\sum_{i=1}^n\left\|W_i\otimes\left[F_i(J)-\Phi(I_t^i|y_i)\right]\right\|^2\tag{4}$$

$\hspace*{2em} W_{i} \in \mathbb{R}_{+}^{H\times W}: \text{per pixel weights}$
$\hspace*{2em} \otimes : \text{Hadamard product}$
$\hspace*{2em}$ Intuitively, the FTD loss reconciles, in the least-squares sense, the different denoising sampling $\hspace*{2em}$ steps, $\Phi(I_{t}^{i}|y_{i})$, suggested on different regions,$F_{i}(J_{t})$, of the generated image $J_{t}$.

![](./imgs/MultiDiffusion_Fig2.png)
Figure 2.MultiDiffusion: a new generation process, $\psi$, is defined over a pre-trained reference model $\Phi$. Starting from a noise image $J_T$, at each generation step, we solve an optimization task whose objective is that each crop $F_i(J_t)$ will follow as closely as possible its denoised version $\Phi(F_i(J_t))$. Note that while each denoising step $\Phi(F_i(J_t))$ may pull to a different direction, our process fuses these inconsistent directions into a global denoising step $\Phi(J_t)$, resulting in a high-quality seamless image.

<center>

![](./imgs/MultiDiffusion_Fig3(a).png)
Figure 3(a). Generation with per-crop independent diffusion paths. As expected, there is no coherency between the crops.
![](./imgs/MultiDiffusion_Fig3(b).png)
Figure 3(b). Generation with fused diffusion paths using MultiDiffusion.

</center>

### Closed-form formula
- In this paper $F_{i}$ consist of direct pixel samples (e.g., taking a crop out of image $J_{t}$). In this case, Eq. 4 is a quadratic Least-Squares (LS) where each pixel of the minimizer $J$ is a weighted average of all its diffusion sample updates, i.e.,
$$\Psi(J_t|z)=\sum_{i=1}^n\frac{F_i^{-1}(W_i)}{\sum_{j=1}^nF_j^{-1}(W_j)}\otimes F_i^{-1}(\Phi(I_t^i|y_i))\tag{5}$$

### Properties of MultiDiffusion
- The main motivation for the deﬁnition of $\psi$ in Eq. 3:

$\hspace*{2em}$ If choose a probability distribution $P_\mathcal{J}$ such that

$$F_i(J_T)\sim P_{\mathcal{I}},\quad\forall i\in[n]\tag{6}$$

$\hspace*{2em}$ and compute $J_{t-1}=\Psi(J_t|z)$, as defined in Eq. 3, where reach a zero FTD loss, $\hspace*{2em} \mathcal{L}_{\mathrm{FTD}}(J_{t-1}|J_{t},z)=0$, then:

$$I_{t-1}^i=F_i(J_t)=\Phi(I_t^i|y_i)$$

$\hspace*{2em}$ That is, $I_t^i$, for all $i\in[n]$, is a diffusion sequence and thus $I_{0}^{i}$ is distributed according to the $\hspace*{2em}$ distribution defined by $\Phi$ over the image space $\mathcal{I}.$

- Summarized as a proposition

$\hspace*{2em}$ If $P_{\mathcal{J}}$ is a distribution over $\mathcal{J}$ satisfying Eq.6,and the FTD cost (Eq.4) is minimized to zero in Eq.3 for all steps $T, T- 1, \ldots, 0$, then the images $I_{t}^{i}= F_{i}( J_{t})$ reproduce a $\Phi$ diffusion path. In particular $F_i(J_0), i\in[n]$ are distributed identically to samples from the reference diffusion model $\phi$.

- The significance of this proposition:

$\hspace*{2em}$ Using a single reference diffusion process we can ﬂexibly adapt to different image generation scenarios without the need to retrain the model, while still being consistent with the reference diffusion model.

### Discussion and Conclusions
- This paper defines a new generation process on top of a pre-trained and fixed diffusion model.
- This approach has several key advantages over previous works:

  - it does not require any further training or finetuning
  - it can be applied to various different generation tasks
  - our generation process yields an optimization task which can be solved in closed form for many tasks, hence can be computed efficiently, while ensuring convergence to the global optimum of our objective

- This approach has a limitation as well:
  
  - our method heavily relies on the generative prior of the reference diffusion model, i.e., the quality of our results depends on the diffusion paths provided by the model