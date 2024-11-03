# SpaText: Spatio-Textual Representation for Controllable Image Generation

## Background
- Task: A text-to-image generation method is proposed, which not only describes the entire scene with a global text prompt but also allows for more precise image generation by using free-form language descriptions for regions of interest.
- Motivation: it is possible to control the shapes of different regions/objects or their layout in a fine-grained fashion than previous attempts.
- Illustration: Lack of fine-grained spatial control: A user with a specific mental image of a Labrador dog holding its paw above a blue ball without touching, can easily generate it with a SpaText representation (left) but will struggle to do so with traditional text-to-image models (right)
  
![](./imgs/SpaText_Fig2.png)

## Contributions
- address a new scenario of image generation with free-form textual scene control
- propose a novel spatio-textual representation that for each segment represents its semantic properties and structure, and demonstrate its effectiveness on two state-of-the-art diffusion models — pixel-based and latent-based
- extend the classifier-free guidance in diffusion models to the multi-conditional case and present an alternative accelerated inference algorithm

## Related Work
- There are no paper about autonomous driving

## Method
### Problem Formulation
- Input
  - $t_{\text{global}}$: a global text prompt 
  - $RST\in\mathbb{R}^{H\times W}$: a raw spatio-textual matrix where $RST[i,j]$ contains the text description of the desired content in pixel $[i, j]$, or $\emptyset$ if the user does not wish to specify the content of this pixel in advance.
- Output: $I\in\mathbb{R}^{3\times H\times W}$
### Overview

![](./imgs/SpaText_Fig3.png)

### CLIP-based Spatio-Textual Representation
- \{$S_i\in[C]\}_{i=1}^K$: randomly chosen disjoint segments 
- $\text{CLIP}_{_{img}}(S_i)$ : crop a tight square around $S_i$ and black-out pixels in the square that are not in the segment. Finally output the CLIP image embedding of the cropped square.
- $ST_{x}\in\mathbb{R}^{H\times W\times d_{\text{CLIP}}}$ : the spatio-textual representation
  $$ST_{x}[j, k] =
    \begin{cases} 
    \text{CLIP}_{\text{img}}(S_{i}) & \text{if } [j, k] \in S_{i} \\
    \vec0 & \text{otherwise}
    \end{cases}$$
- During inference
  - Only text description and a spatial map are available
  - Leverage the common embedding space of text and image extracted by CLIP
  - To further mitigate the domain gap, a prior model is trained separately to convert CLIP text embeddings to CLIP image embeddings
  
### Incorporating Spatio-Textual Representation into SoTA Diffusion Models
Authors compared pixel-based (DALL·E-2) and latent-based (Stable Diffusion) models
1. DALL·E 2
   - Conditioning: Concatenate $x_t$ and $ST$ (from $(C_{in},K_H,K_W)$ to $(C_{\text{in}} + d_{\text{CLIP}}, K_H, K_W)$) in order to keep the spatial correspondence between the spatio-textual representation ST and the noisy image $x_{_{t}}$ at each stage

       $$L_{\text{simple}} = E_{t, x_0, \epsilon} \left[ \left|\left| \epsilon - \epsilon_{\theta}(x_t, \mathrm{CLIP}_{\mathrm{img}}(x_0), ST, t) \right|\right|^2 \right]$$

     -  $\epsilon_{\theta}$ : a UNet model that predicts the added noise
     -  $t$ : time step
     -  $x_{_{t}}$ : the noisy image at time step t
     -  $ST$ : spatio-textual representation

2. Stable Diffusion (latent-based model)
   - Conditioning: Concatenation
   - An autoencoder $(Enc(x),Dec(z))$ that embeds the image $x$ into a lower-dimensional latent space $z$
   - A diffusion model $A$ that performs the following denoising steps on the latent space $z_{t-1}=A(z_{t},\mathrm{CLIP_{txt}}(t))$
   - The final denoised latent is fed to the decoder to get the final prediction $Dec(z_{0})$

        $$L_{\mathrm{LDM}} = E_{t,y,z_{0},\epsilon}\left[ \left|\left| \epsilon-\epsilon_{\theta}(z_{t},\mathrm{CLIP}_{\mathrm{txt}}(y),ST,t) \right|\right|^{2}\right]$$
        - $z_{_{t}}$ : the noisy latent code at time step t
        - $y$ : the corresponding text prompt

### Multi Conditional Classifier-Free Guidence
- Extend Classifier-Free Guidence to support multiple conditions

    $$
    \hat\epsilon_{\theta} (x_t|\{c_i\}_{i=1}^N) = \epsilon_{\theta} (x_t|\emptyset) + \sum_{i=1}^N s_i \Delta_i^t
    $$

  - $\{c_{i}\}_{i=1}^{i=N}$ : N condition inputs
  - $\epsilon_{\theta}(x_{t}|\emptyset)$ : during training replace each condition $c_{_{i}}$ with the null condition $\boldsymbol{\phi}$
  - $\Delta_{i}^{t}=\epsilon_{\theta}(x_{t}|c_{i})-\epsilon_{\theta}(x_{t}|\emptyset)$ : during inference ,extrapolate towards the direction of the condition $\epsilon_{\theta}(x_{t}|c)$ and away from $\epsilon_{\theta}(x_{t}|\emptyset)$
  - $s_{_{i}}$ : N guidance scales, each $\geq1$

  Using the above formulation allows fine-grained control over the input conditions
  
  ![](./imgs/SpaText_Fig4.png)

Illustration: Given the same inputs (left) ,we can use different scales for each condition.In this example,if we put all the weight on the local scene (1),the generated image contains a horse with the correct color and posture,but not at the beach.

## Limitations
- When there are more than a few segments, the model might miss some of the segments or propagate their characteristics
- The model struggles to handle tiny segments
- The model can only address the scenario of text-to-image generation with sparse scene control
