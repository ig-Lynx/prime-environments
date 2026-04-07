from __future__ import annotations

from typing import Any

import verifiers as vf
from datasets import Dataset
from verifiers.parsers.parser import Parser


class PassthroughParser(Parser):
    def parse(self, text: str | None) -> str | None:
        if text is None:
            return None
        return str(text)


def load_environment(**kwargs: Any) -> vf.Environment:
    dataset = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": ""}]],
            "answer": [""],
            "task": ["Pydantic"],
        }
    )

    parser = PassthroughParser()
    rubric = vf.Rubric(funcs=[lambda **_: 0.0], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(dataset=dataset, parser=parser, rubric=rubric, **kwargs)
