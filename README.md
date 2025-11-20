# Recreation of KnowNo
Through this project, I attempt to recreate the results of [KnowNo](https://github.com/google-research/google-research/tree/master/language_model_uncertainty), primarily the statistical data, and ideally tesing on hardware down the line. The goal of this is primarily for learning purposes, specifically on gaining more knowledge on the interaction of LLMs on real world systems, conformal prediction, & overall AI research in general.

## Overview
KnowNo serves as a framework that measures and aligns the uncertainty of LLMs regarding choosing the best outcome given a task. When stuck on a decision (often due to ambiguity), the LLM will then prompt the human for help in order to complete its task. Through the usage of conformal prediciton (CP), the LLM provides a statisical gurantee that the answers it provides must atleast meet a certain threshold in order to be considered for the prediction set. Although the original paper consisted of 3 experiments (Mobile Manipulation, Table Rearrangement, & Bimanual Manipulation), they all follow the same concept of having the LLM form a 4 answer multiple choice question and forming the prediction set. If the prediction set has more than one answer, then human help is prompted, if not, execute the action.

## Experimental Data (Software Only)

### Tabletop Rearrangement

<details>
<summary><strong>Show Tabletop Rearrangement results</strong></summary>

<br>

![Table 1](/knowno-recreation/images/tabletop_rearrangement_KnowNo.png)  

vs.  

![KnowNo's Table1](/KnowNo's%20Table1.png)

</details>

### Mobile Manipulation

<details>
<summary><strong>Show Mobile Manipulation results</strong></summary>

<br>

![Table 2](/knowno-recreation/images/mobile_manipulation_KnowNo.png)  

vs.  

![KnowNo's Table2](/KnowNo's%20Table2.png)

</details>

## Deviations from Paper
There were some deviations from the original paper that have defintiley affected results. Despite this, I feel that the experiment was still a success as a lot of datasets were more accurate and led me to a better overall understanding for LLMs. These include:
- Uses **Ollama Llama 3.1:8b** instead of **PaLM-2L** or **GPT 3.5**, necessary change as PaLM-2L has been discontinued and GPT 3.5 requires token purchases, but Ollama 3.1:8b has better capabilities

  <details>
  <summary><strong>Show Ollama vs GPT comparison</strong></summary>

  <br>

  ![Ollama vs. GPT](/knowno-recreation/images/Ollama%203.1%20vs.%20gpt%203.5.png)

  </details>

- Smaller test set, while the original paper's Mobile Manipulation test set had **270** tests, I only performed **100** due to hardware restrictions (ran on laptop)
- Initially, I wanted to generate my own calibration results but this caused the CP prediction sets to vary too far, so I used the given 1 - ϵ provided. This is allowed from the paper. Code snippets for calibration can be found in  
  > "The calibration invoves running LLM inference for a fair number of calibration data (200). You can also skip the calibration and directly test in new scenarios at the last block based on provided calibration result."
- No interaction with robots in actual environments, hardware restrictions
- Didn't account for entire plans, just tasks
  
## Challenges
- Learning - Most of the time was spent making sense of the paper, key algorithms and how they worked, how to code what I needed, core techstack & libraries, fixing bugs, etc.
- Jupyter notebooks not working: [Mobile Manipulation in the Github](https://github.com/google-research/google-research/blob/master/language_model_uncertainty/KnowNo_MobileManipulation.ipynb) & [Tabletop Sim](https://github.com/google-research/google-research/blob/master/language_model_uncertainty/KnowNo_TabletopSim.ipynb)
- Finding an open source/free LLM that supported log probabilites, thankfully Ollama's v0.12.11 (released barely last week) had been implemented to support top log probabilites
- Working around large datasets with limited hardware
- Proper steps to take when calibrating/experimenting/evaluating

## Conclusion/My Takeaways
KnowNo served as a great baseline approach to combatting LLM uncertainty and hallucinations through the usage of CP, and I'm sure that it served as a crucial building block for LLM optimizations in efficency and autonomy. I am sure that when KnowNo was published, and even to now, many people were able to take advantage of KnowNo's results to grow LLM capabilities. The goal of recreating exact results of KnowNo did not go as planned due to many deviations including models used, hardware restrictions, and limited capabilities. I'm sure that Despite this, I have greatly expanded my knowledge on LLMs, tech stacks, conformal prediction, and reserach as a whole. If I were to do this again, I plan to further develop my code and ideally stay more true to the paper as I'm aware this would never work as a proper recreation of a scientific paper.
