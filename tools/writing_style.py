"""
tools/writing_style.py
──────────────────────
Anti-AI-writing-tell instructions injected into prose-generating LLM prompts.

Two variants:
  ANTI_AI_TELL_INSTRUCTION          — for all formal prose outputs
  ANTI_AI_TELL_NARRATIVE_INSTRUCTION — for spoken/audio/podcast content
                                       (allows natural spoken transitions)

JSON-output prompts (FAQ, mind maps, knowledge graphs, screening, PICO, RoB,
GRADE) must NOT include these instructions — they would break structured output
parsing. Only inject into prompts that produce user-facing prose.
"""

ANTI_AI_TELL_INSTRUCTION = """
WRITING STYLE — AVOID AI WRITING TELLS:
• Forbidden words: delve, tapestry, testament, groundbreaking, meticulous, intricate,
  bustling, vibrant, pivotal, comprehensive, multifaceted, nuanced, robust, leveraging,
  spearheading, foster, elevate, streamline, paramount, synergy, paradigm shift,
  game-changer, cutting-edge, revolutionary, transformative, holistic, dynamic.
• Never open sentences with: "Certainly!", "Of course!", "Absolutely!", "Great!",
  "Sure!", "Notably,", "It is worth noting that", "It is important to note that".
• Never start paragraphs with: "In conclusion,", "To summarize,", "Furthermore,",
  "Moreover,", "In addition,", "Additionally," as the very first word of a section.
• Avoid hollow hedging: "it could be argued", "one might suggest", "it seems that"
  when a direct claim is supported by the sources.
• Do not restate the question before answering it.
• Write varied sentence lengths. Mix short punchy sentences with longer analytical ones.
• Prefer specific, concrete language over vague generalities.
"""

ANTI_AI_TELL_NARRATIVE_INSTRUCTION = """
WRITING STYLE — AVOID AI WRITING TELLS:
• Forbidden words: delve, tapestry, testament, groundbreaking, meticulous, intricate,
  bustling, vibrant, pivotal, comprehensive, multifaceted, nuanced, robust, leveraging,
  spearheading, foster, elevate, streamline, paramount, synergy, paradigm shift,
  game-changer, cutting-edge, revolutionary, transformative, holistic, dynamic.
• Never open with: "Certainly!", "Of course!", "Absolutely!", "Great!", "Sure!".
• Avoid hollow hedging and vague generalities — be specific and concrete.
• Write with a natural spoken rhythm. Vary sentence length. Short sentences land.
• Natural spoken transitions (First, Then, Next, Finally) are fine for audio flow.
• Do not restate the question or topic before addressing it.
"""
