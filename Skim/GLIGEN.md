# GLIGEN: Open-Set Grounded Text-to-Image Generation

## Background
**Task**: Proposing a novel approach called GLIGEN, which achieves open-world grounded text2img generation with caption and bounding box condition inputs, and the grounding ability generalizes well to novel spatial configurations and concepts.

**The drawback of existing methods**: The existing large-scale text-to-image generation models cannot be conditioned on other input modalities apart from text, and thus lack the ability to precisely localize concepts, use reference images, or other conditional inputs to control the generation process. For example, it is difficult to describe the precise location of an object using text, whereas bounding boxes / keypoints can easily achieve this, as shown in Figure.

**Illustration**: 
- The left image: Through these two colored rectangular boxes, the model can identify and locate specific areas in the image. 
- Caption: The text input for generating the image is "Elon Musk and Emma Watson on a movie poster".
- Grounded text: The specific characters in the generated image, "Elon Musk" and "Emma Watson", are indicated in red and green, matching the colors of the rectangular boxes above. 
- Grounded style image: The style information used for generating the image is marked, with style hints shown through the blue inset box.

![](./imgs/GLIGEN_Fig1.png)

**Idea**: 
- Providing new grounding conditional inputs to pretrained text-to-image diffusion models. 
- Preserving the original vast concept knowledge in the pretrained model while learning to inject the new grounding information. 
- To prevent knowledge forgetting, freezing the original model weights and add new trainable gated Transformer layers that take in the new grounding input (e.g., bounding box). 
- To enable model to ground open-world vocabulary concepts, using the same pre-trained text encoder (for encoding the caption) to encode each phrase associated with each grounded entity (i.e., one phrase per bounding box) and feeding the encoded tokens into the newly inserted layers with their encoded location information.
- To further improve the model’s grounding ability, we unify the object detection and grounding data formats for training, following GLIP.

**Advantage**: 
- The model can generalize to unseen objects even when only trained on the COCO dataset.
- Its generalization on LVIS outperforms a strong fully-supervised baseline by a large margin.

## Contributions
- Propose a new text2img generation method that endows new grounding controllability over existing text2img diffusion models
- By preserving the pre-trained weights and learning to gradually integrate the new localization layers, the model achieves open-world grounded text2img generation with bounding box inputs, i.e., synthesis of novel localized concepts unobserved in training
- The model’s zero-shot performance on layout2img tasks significantly outperforms the prior state-of-the-art, demonstrating the power of building upon large pretrained generative models for downstream tasks.

## Related Work
- There are no paper about autonomous driving

## Method
### Preliminaries on Latent Diffusion Models
- focus on the latent generation space of LDM (Latent Diffusion Model)

#### Training Objective
$$\min_{\boldsymbol{\theta}}\mathcal{L}_{\mathrm{LDM}}=\mathbb{E}_{\boldsymbol{z},\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),t}\big[\|\boldsymbol{\epsilon}-f_{\boldsymbol{\theta}}(\boldsymbol{z}_t,t,\boldsymbol{c})\|_2^2\big]\tag{1}$$
- - $\boldsymbol{z}_{t}$: the step-$t$ noisy variant of input $z$
  - $f_{\boldsymbol{\theta}}(*,t,\boldsymbol{c})$: The model that reduces noise based on time steps and descriptive information.
  - $t$: uniformly sampled from time steps $\{1,\cdots,T\}$
  - $\boldsymbol{c}$: caption

#### Network Architecture
- The caption feature $c$ is encoded via a fixed CLIP text encoder in Stable Diffusion and the time $t$ is first mapped to time embedding $\phi(t)$, then injected into the UNet.
- Denoising Autoencoder $f_{\boldsymbol{\theta}}(*,t,\boldsymbol{c})$ is implemented via UNet. It takes in a noisy latent $z$, as well as information from time step $t$ and condition $c$. It consists of a series of ResNet and Transformer blocks. The caption feature is used in a cross attention layer within each Transformer block.

### Open-set Grounded Image Generation
#### I.Grounding Instruction Input
- Denote the semantic information of the grounding entity as $e$, which can be described either through text or an example image; and as $l$ the grounding spatial configuration described with e.g., a bounding box, a set of keypoints, or an edge map, etc.
- Define the instruction to a grounded text-to-image model as a composition of the caption and grounded entities:
$$\text{Instruction:} \ \boldsymbol{y}=(\boldsymbol{c},\boldsymbol{e}) \text{, with} \tag{2}$$

  $$\text{Caption:} \ \boldsymbol{c}=[c_1,\cdots,c_L]\tag{3}$$

  $$\text{Grounding:} \ \boldsymbol{e}=[(e_1,\boldsymbol{l}_1),\cdots,(e_N,\boldsymbol{l}_N)]\tag{4}$$
  - $L$: the caption length
  - $N$: the number of entities to ground
  - For the grounded entity $e$, mainly focus on using text as its representation due to simplicity.

![](./imgs/GLIGEN_Fig2.png)
$$\text{Illustration: Grounding token construction process for the bounding box with text case}$$

##### Caption Tokens
- The caption $c$ is processed in the same way as in LDM. Specifically, obtain the caption feature sequence (yellow tokens in Figure) using $h^c=[h_1^c,\cdots,h_L^c]=f_{\mathrm{text}}(\boldsymbol{c})$, where $h_\ell^c$ is the contextualized text feature for the $\ell$-th word in the caption.

##### Grounding Tokens
- For each grounded text entity denoted with a bounding box, represent the location information as $\boldsymbol{l}=[\alpha_{\min},\beta_{\min},\alpha_{\max},\beta_{\max}]$ with its top-left and bottom-right coordinates.
- For the text entity $e$, use the same pretrained text encoder to obtain its text feature $f_{\mathrm{text}}(e)$ (light green token in Figure), and then fuse it with its bounding box information to produce a grounding token (dark green token in Figure).
$$h^e=\mathrm{MLP}(f_{\mathrm{text}}(e),\mathrm{Fourier}(\boldsymbol{l}))\tag{5}$$
  - $\mathrm{Fourier}$: the Fourier embedding 
  - $\mathrm{MLP(⋅, ⋅)}$: a multi-layer perceptron that concatenates the two inputs across the feature dimension.
  - The grounding token sequence is represented as $\boldsymbol{h}^e=[h_1^e,\cdots,h_N^e]$

##### From Closed-set to Open-set
- Desigen transfers from Closed-set to Open-set
- The existing layout2img works only deal with a closed-set setting (e.g., COCO categories), and thus the model can only ground the observed entities in the generated images, lacking the ability to generalize to ground new entities.
- In the open-set design, since the noun entities are processed by the same text encoder as used to encode the caption, so even when the localization information is limited to the concepts in the grounding training datasets, the model can still generalize to other concepts.
- The closed-set models can only be tested on known categories from the training data, evaluating generation quality and grounding accuracy. In contrast, the open-set models can generalize to unseen categories, demonstrating strong performance in zero-shot tasks during inference, especially for rare categories.

#### II. Continual Learning  for Grounded Generation
- Endow new spatial grounding capabilities to existing large language-to-image generation models.
- Retain the previous knowledge in the model weights while expanding the new capability. Lock the original model weights, and gradually adapt the model by tuning new modules.

##### Gated Self-Attention
![](./imgs/GLIGEN_Fig3.png)
$$\text{The purple box: Visual}$$

$$\text{The yellow box: Caption}$$
  
$$\text{The green box: Grounding}$$

- The original Transformer block of LDM consists of two attention layers: The self-attention over the visual tokens, followed by cross-attention from caption tokens. By considering the residual connection, the two layers can be written:
$$\boldsymbol{v}=\boldsymbol{v}+\text{SelfAttn}(v)\tag{6}$$

  $$\boldsymbol{v}=\boldsymbol{v}+\operatorname{CrossAttn}(\boldsymbol{\upsilon},\boldsymbol{h}^c)\tag{7}$$
  - $\boldsymbol{v}$: the visual feature tokens of an image $\boldsymbol{v} = [v_1,\cdots,v_M]$
- Freeze these two attention layers and add a new gated self-attention layer to enable the spatial grounding ability; see Figure.
$$\boldsymbol{v}=\boldsymbol{v}+\beta\cdot\tanh(\gamma)\cdot\mathrm{TS}(\mathrm{SelfAttn}([\boldsymbol{v},\boldsymbol{h}^e]))\tag{8}$$
  - the attention is performed over the concatenation of visual and grounding tokens $[\boldsymbol{\upsilon},\boldsymbol{h}^e]$
  - $\mathit{TS}(\cdot)$: a token selection operation that considers visual tokens only 
  - $\gamma$: a learnable scalar which is initialized as 0 ($\gamma$ is initialized at 0, meaning that the model is not influenced by the grounding information at the beginning. As training progresses, $\gamma$ can be gradually adjusted, allowing the influence of the grounding information to increase progressively.)
  - $\beta$: is set as 1 during the entire training process and is only varied for scheduled sampling during inference for improved quality and controllability.($\beta$ is fixed at 1 to ensure that the influence of the grounding information remains constant during training.)



##### Learning Procedure
$$\mathit{min}_{\boldsymbol{\theta}^{\prime}}\mathcal{L}_{\mathrm{Grounding}}=\mathbb{E}_{\boldsymbol{z},\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),t}{\left[\left\|\boldsymbol{\epsilon}-f_{\{\boldsymbol{\theta},\boldsymbol{\theta}^{\prime}\}}(\boldsymbol{z}_t,t,\boldsymbol{y})\right\|_2^2\right]}\tag{9}$$
- - $\mathbf{\theta}$: the parameters of the original model, which remain unchanged.
  - $\mathbf{\theta}^{\prime}$: the new parameters, used to adapt to the new grounding information.
  - $y$: the grounding information input, which may include bounding boxes, keypoints, or other grounding data.

##### Scheduled Sampling in Inference
- This constant $\beta$ sampling scheme provides overall good performance in terms of both generation and grounding, but sometimes generates lower quality images compared with the original text2img models. To strike a better trade-off between generation and grounding for GLIGEN, a scheduled sampling scheme is proposed.
- There is flexibility during inference to schedule the diffusion process to either use both the grounding and language tokens or use only the language tokens of the original model at anytime, by setting different $\beta$ values.
$$\beta=\begin{cases}1,&t\leq\tau*T&\#\text{Grounded inference stage}\\0,&t>\tau*T&\#\text{Standard inference stage}&\end{cases}\tag{10}$$
  - Introduce a two-stage inference procedure, divided by $\tau\in[0,1]$. For a diffusion process with $T$ steps, one can set $\beta$ to 1 at the first $\tau*T$ steps, and set $\beta$ to 0 for the remaining $(1-\tau)*T$ steps.
- The major benefit of scheduled sampling is improved visual quality as the rough concept location and outline are decided in the early stages, followed by fine-grained details in later stages.

