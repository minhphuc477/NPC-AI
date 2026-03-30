"""Annotation guidelines module.

Provides AnnotationGuidelines class that can generate markdown guidelines and
save JSON/markdown artifacts for annotators.
"""
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Dict, Any
import json
import os
from pathlib import Path


class CharacterRole(Enum):
    """Enumerates a few example character roles used in the NPC dialogues."""
    GATEKEEPER = "Gatekeeper"
    HERO = "Hero"
    MERCHANT = "Merchant"


@dataclass
class CharacterProfile:
    """Represents a character's annotation-relevant traits."""
    name: str
    role: CharacterRole
    description: str


@dataclass
class DialogueExample:
    """Example of a dialogue turn and preferred annotation."""
    context: str
    utterance: str
    annotation: Dict[str, Any]


class AnnotationGuidelines:
    """Generate annotation guidelines in markdown and JSON.

    Methods
    -------
    create_markdown_guidelines()
        Return a markdown string describing guidelines.
    save_guidelines(output_dir='guidelines')
        Save guidelines as JSON + markdown and a quick reference file.
    _create_quick_reference()
        Internal helper returning a short quick ref markdown string.
    """

    def __init__(self, title: str = "NPC Dialogue Annotation Guidelines"):
        self.title = title
        self.character_profiles: List[CharacterProfile] = [
            CharacterProfile(
                name="Gatekeeper",
                role=CharacterRole.GATEKEEPER,
                description=(
                    "A cautious NPC who controls access; stern but fair. "
                    "Annotate directives and refusal acts carefully."
                ),
            ),
            CharacterProfile(
                name="Player",
                role=CharacterRole.HERO,
                description="The player protagonist; may request access or bribe.",
            ),
        ]
        self.guidelines: List[str] = [
            "Label each utterance with one or more dialogue acts (e.g., request, inform, refuse, accept).",
            "When ambiguous, prefer the more conservative label and add a comment.",
            "Include contextual notes when the intent is implicit (e.g., sarcasm or oblique refusal).",
        ]
        self.examples: List[DialogueExample] = [
            DialogueExample(
                context="At a city gate, player asks to enter without ticket.",
                utterance="You can't enter without a permit.",
                annotation={"acts": ["refuse"], "notes": "Gatekeeper enforces rules."},
            ),
            DialogueExample(
                context="Merchant responding to a bargaining attempt.",
                utterance="I can lower the price if you buy two.",
                annotation={"acts": ["offer", "inform"], "notes": "Implicit negotiation."},
            ),
        ]

    def create_markdown_guidelines(self) -> str:
        """Return the full guidelines as a markdown string."""
        lines: List[str] = [f"# {self.title}", ""]
        lines.append("## Character Profiles")
        for cp in self.character_profiles:
            lines.append(f"### {cp.name} ({cp.role.value})")
            lines.append(cp.description)
            lines.append("")
        lines.append("## Guidelines")
        for i, g in enumerate(self.guidelines, start=1):
            lines.append(f"{i}. {g}")
        lines.append("")
        lines.append("## Examples")
        for ex in self.examples:
            lines.append(f"- **Context**: {ex.context}")
            lines.append(f"  - **Utterance**: {ex.utterance}")
            lines.append(f"  - **Annotation**: {json.dumps(ex.annotation, ensure_ascii=False)}")
            lines.append("")
        lines.append("---")
        lines.append(self._create_quick_reference())
        return "\n".join(lines)

    def _create_quick_reference(self) -> str:
        """Private helper returning a short quick reference as markdown."""
        quick = ["## Quick Reference", "- Use `refuse` when entry is denied.", "- Use `offer` for alternative suggestions."]
        return "\n".join(quick)

    def save_guidelines(self, output_dir: str = "guidelines") -> Path:
        """Save guidelines as JSON and markdown files and return the base path.

        JSON will include character profiles, guideline list, and examples. Uses
        ensure_ascii=False to preserve non-ascii characters.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        data = {
            "title": self.title,
            "character_profiles": [
                {"name": cp.name, "role": cp.role.value, "description": cp.description}
                for cp in self.character_profiles
            ],
            "guidelines": self.guidelines,
            "examples": [asdict(ex) for ex in self.examples],
        }

        json_path = out / "guidelines.json"
        md_path = out / "guidelines.md"
        quick_path = out / "quick_reference.md"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        md_content = self.create_markdown_guidelines()
        with md_path.open("w", encoding="utf-8") as f:
            f.write(md_content)

        with quick_path.open("w", encoding="utf-8") as f:
            f.write(self._create_quick_reference())

        return out


if __name__ == "__main__":
    # Simple CLI: create artifacts and print the created path
    out = AnnotationGuidelines().save_guidelines()
    print(f"Saved guidelines to: {out}")
    # Exit code 0 on success
