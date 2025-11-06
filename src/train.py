import copy
import json
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()


class TrainingFreeGRPO:
    def __init__(self, max_workers: int = 6, temperature: float = 1.0) -> None:
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
        )
        self.max_workers = max_workers
        self.temperature = temperature

    def get_llm_output(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL"),
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def get_llm_output_batch(self, prompts: list[str]) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self.get_llm_output, prompts))

    def load_experiences(self, experiences_file: str) -> dict[str, str]:
        if os.path.exists(experiences_file):
            with open(experiences_file, "r", encoding="utf-8") as f:
                experiences = json.load(f)
                return experiences
        return {}

    def train(
        self,
        train_data_file: str,
        experiences_file: str,
        batch_size: int = 2,
        epochs: int = 3,
        n: int = 4,
    ) -> None:
        from prompts import (
            EXPERIENCE_GENERATE_PROMPT,
            PROBLEM_WITH_EXPERIENCE_PROMPT,
            BATCH_EXPERIENCE_UPDATE_PROMPT,
        )

        train_data = json.load(open(train_data_file, "r", encoding="utf-8"))

        steps = len(train_data) * epochs // batch_size + 1
        pbar = tqdm(total=steps, desc="Training Progress")

        for epoch in range(epochs):
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i : i + batch_size]
                experiences = self.load_experiences(experiences_file)
                prompts = []

                # 准备测试数据
                for item in batch:
                    item_prompts = [
                        PROBLEM_WITH_EXPERIENCE_PROMPT.format(
                            problem=item["prompt"], experiences=experiences
                        )
                    ] * n
                    prompts.extend(item_prompts)

                # 获取llm结果
                llm_outputs = self.get_llm_output_batch(prompts)

                # 整理结果
                candidate_experiences = copy.deepcopy(experiences)
                to_modify = []
                for idx, item in enumerate(batch):
                    start = idx * n
                    end = start + n
                    trajectory_texts = llm_outputs[start:end]

                    experiences_prompt = EXPERIENCE_GENERATE_PROMPT.format(
                        problem=item["prompt"],
                        trajectories="\n\n".join(trajectory_texts),
                        answer=item["answer"],
                        experiences=experiences,
                        max_operations=1,
                    )

                    # 生成经验
                    single_sample_experience = self.get_llm_output(experiences_prompt)
                    single_sample_experience = (
                        single_sample_experience.split("```json")[-1]
                        .split("```")[0]
                        .strip()
                    )

                    try:
                        single_sample_experience = json.loads(single_sample_experience)
                        for exp in single_sample_experience:
                            if exp["option"] == "add":
                                candidate_experiences[
                                    str(len(candidate_experiences) + 1)
                                ] = exp["experience"]
                            elif exp["option"] == "modify":
                                if str(exp["modified_from"]) in candidate_experiences:
                                    to_modify.append(exp)

                    except Exception as e:
                        print(f"Error parsing experience output: {e}")
                        continue

                # 更新经验
                update_experiences_prompt = BATCH_EXPERIENCE_UPDATE_PROMPT.format(
                    experiences=candidate_experiences, updates=to_modify
                )
                update_experiences = self.get_llm_output(update_experiences_prompt)
                update_experiences = (
                    update_experiences.split("```json")[-1].split("```")[0].strip()
                )

                pbar.update(1)

                new_experiences = copy.deepcopy(candidate_experiences)
                try:
                    update_experiences = json.loads(update_experiences)
                    max_id = max([int(k) for k in new_experiences.keys()], default=0)

                    for exp in update_experiences:
                        if exp["option"] == "modify":
                            if str(exp["modified_from"]) in new_experiences:
                                new_experiences[str(exp["modified_from"])] = exp[
                                    "experience"
                                ]
                        elif exp["option"] == "merge":
                            to_merge = exp.get("merged_from", [])
                            for exp_id in to_merge:
                                if str(exp_id) in new_experiences:
                                    del new_experiences[str(exp_id)]
                            new_experiences[str(max_id + 1)] = exp["experience"]
                            max_id += 1
                    new_experiences = {
                        str(idx + 1): exp
                        for idx, exp in enumerate(new_experiences.values())
                    }

                    with open(experiences_file, "w", encoding="utf-8") as f:
                        json.dump(new_experiences, f, ensure_ascii=False, indent=4)

                except Exception as e:
                    print(f"Error parsing final experience updates: {e}")
                    continue


def main():

    trainer = TrainingFreeGRPO(max_workers=5, temperature=1.0)

    # 生成经验
    trainer.train(
        train_data_file="data/train_data.json",
        experiences_file="data/experiences.json",
        batch_size=4,
        epochs=3,
        n=4,
    )


if __name__ == "__main__":
    main()
