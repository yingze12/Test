# Review Process

1. Checkout to a new branch with the following naming convention: `date_name`, e.g. `20241014_kaiwen`.
2. Go to the papers you are assigned this week.
3. For each paper, follow the review process below.

   1. Read the title, abstract, introduction, and conclusion.
   2. Ask yourself the following questions:
      - **Is this paper tailored for autonomous driving?**
        - Yes/No
      - **Is this paper related to diffusion models?**
        - Yes/No
   3. 
      - If there are 2 "Yes" above: In-depth review necessary, follow instructions marked with <kbd style="background-color: orange;">In-depth</kbd>.
      - If there is only 1 "Yes": Skim through the paper, follow instructions marked with <kbd style="background-color: cyan;">Skim</kbd>.
      - Otherwise: Go to 10.

   4. <kbd style="background-color: orange;">In-depth</kbd> <kbd style="background-color: cyan;">Skim</kbd> Summerize the background / goal.
       - Is this paper designed to solve a specific task? 
         - If yes
           - What is the task?
           - Is it a novel task? If yes, what is the motivation? If no, what is the drawback of existing methods? 
         - If no
           - What is the motivation of this paper? Training with less resources, faster inference, explainability, novel architecture, etc.
           - Why is this challenging?
   5. <kbd style="background-color: orange;">In-depth</kbd> List all contributions of the paper, which are typically already summerized in the Introduction.
   6. <kbd style="background-color: orange;">In-depth</kbd><kbd style="background-color: cyan;">Skim</kbd> If the paper is related to autonomous driving, go through the related work and collect useful papers with links.
   7.  <kbd style="background-color: orange;">In-depth</kbd> <kbd style="background-color: cyan;">Skim</kbd> Explain the method.
       - Select a figure from the paper that best explains the method.
       - What is the proposed method?
       - What is the intuition behind the proposed method?
       - What are the key components of the proposed method?
       - What are the key equations of the proposed method?: Use LaTeX to write the equations, e.g. https://lukas-blecher.github.io/LaTeX-OCR/!
   8.  <kbd style="background-color: orange;">In-depth</kbd> How are the experiments conducted?
       - What are the datasets used?
       - What are the evaluation metrics?
   9.  <kbd style="background-color: orange;">In-depth</kbd> <kbd style="background-color: cyan;">Skim</kbd> Document your reviews, name the corresponding markdown after the paper's title and save it to either `In-depth` or `Skim`. Also add the paper title and a one-sentence summary (task + improvement) to the corresponding section in the README.md. 
   10. Mark the paper as reviewed in the [README.md](README.md).
4. Create a pull request for this branch and pin me.
