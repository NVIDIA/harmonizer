## Description: <br>
Use to run NVIDIA DiffusionHarmonizer (public successor to the older Fixer recipes) to enhance, harmonize, evaluate, or fine-tune novel-view frames from NuRec / 3DGS / NeRF reconstructions. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers who need to enhance, harmonize, or evaluate rendered frames from neural reconstruction pipelines (NuRec, 3DGS, NeRF) using NVIDIA DiffusionHarmonizer. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Inference and Container Build Guide](references/inference.md) <br>
- [Evaluation and Metrics Guide](references/evaluation.md) <br>
- [Training and Fine-Tuning Guide](references/training.md) <br>
- [Wrapper Image Build Guide](references/wrapper-image.md) <br>
- [Troubleshooting Guide](references/troubleshooting.md) <br>
- [Teardown and Cleanup Guide](references/teardown.md) <br>
- [Harmonizer Source Code](https://github.com/NVIDIA/harmonizer) <br>
- [Harmonizer Model Card](https://huggingface.co/nvidia/Harmonizer) <br>
- [Harmonizer Dataset](https://huggingface.co/datasets/nvidia/Harmonizer-Dataset) <br>
- [DiffusionHarmonizer Paper (arXiv 2602.24096)](https://arxiv.org/abs/2602.24096) <br>
- [DiffusionHarmonizer Project Page](https://research.nvidia.com/labs/sil/projects/diffusion-harmonizer/) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Tasks: <br>
Not yet evaluated in this repository. The skill was evaluated through NVSkills-Eval before it was decoupled and moved here; that run predates the current content, so its verdict is not carried over. Re-run NVSkills-Eval against this copy. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>



## Skill Version(s): <br>
0.5.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
