# Recreation of KnowNo
Through this project, I attempt to recreate the results of [KnowNo](https://github.com/google-research/google-research/tree/master/language_model_uncertainty), primarily the statistical data, and ideally tesing on hardware down the line. The goal of this is primarily for learning purposes, specifically on gaining more knowledge on the interaction of LLMs on real world systems, conformal prediction, & overall AI research in general.

## Overview
KnowNo serves as a framework that measures and aligns the uncertainty of LLMs regarding choosing the best outcome given a task. When stuck on a decision (often due to ambiguity), the LLM will then prompt the human for help in order to complete its task. Through the usage of conformal prediciton (CP), the LLM provides a statisical gurantee that the answers it provides must atleast meet a certain threshold in order to be considered for the prediction set. Although the original paper consisted of 3 experiments (Mobile Manipulation, Table Rearrangement, & Bimanual Manipulation), they all follow the same concept of having the LLM form a 4 answer multiple choice question and forming the prediction set. If the prediction set has more than one answer, then human help is prompted, if not, execute the action.

## Experimental Data (Software Only)
### Mobile Manipulation



## Deviations from Paper
There were some deviations from the original paper that have defintiley affected results. Despite this, I feel that the experiment was still a success as a lot of datasets were more accurate and led me to a better overall understanding for LLMs. These include:
- Uses **Ollama Llama 3.1:8b** instead of **PaLM-2L** or **GPT 3.5**, necessary change as PaLM-2L has been discontinued and GPT 3.5 requires token purchases
  - However, Ollama Llama 3.1:8b has better capabilities than GPT 3.5
- Smaller test set, while the original paper's Mobile Manipulation test set had **270** tests, I only performed **100** due to hardware restrictions (ran on laptop)
- Initially, I wanted to generate my own calibration results but this caused the CP prediction sets to vary too far, so I used the given 1 - ϵ provided. This is allowed from the paper. Code snippets for calibration can be found in
>        " The calibration invoves running LLM inference for a fair number of calibration data (200). You can also skip the calibration and directly test in new scenarios at the last block based on provided calibration result."
- No interaction with robots in actual environments, hardware restrictions

