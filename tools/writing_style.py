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

ANTI_AI_TELL_REVIEWER_INSTRUCTION = """
WRITING STYLE — AVOID AI WRITING TELLS (Wikipedia Signs of AI Writing):

Forbidden vocabulary — never use these words:
  delve, tapestry, testament, groundbreaking, meticulous, intricate, bustling, vibrant,
  pivotal, comprehensive, multifaceted, nuanced, robust, leveraging, spearheading, foster,
  elevate, streamline, paramount, synergy, paradigm shift, game-changer, cutting-edge,
  revolutionary, transformative, holistic, dynamic, underscore, crucial, enhance, landscape,
  realm, interplay, garnered, bolstered, enduring, impactful, innovative, significant
  (as a vague intensifier), key (as an adjective meaning "important").

Forbidden sentence and paragraph openers:
  "In conclusion,", "In summary,", "Overall,", "To summarize,", "Furthermore,",
  "Moreover,", "Additionally,", "Notably,", "It is worth noting that",
  "It is important to note that", "It is worth remembering that",
  "No discussion would be complete without", "Certainly!", "Of course!", "Absolutely!".

Forbidden structural habits (Wikipedia-documented AI patterns):
  • Compliment sandwich: do NOT wrap criticism between praise — reviewers must state
    weaknesses directly, without a softening preamble or a positive conclusion.
  • Hourglass structure: do NOT open and close with generic synthesis paragraphs.
    Start immediately with the substance of the first section.
  • "Not X, but Y" manufactured contrast: avoid formulas like "Not merely X, but Y"
    or "Less X than Y" when a direct statement suffices.
  • "Faces challenges like…; despite these challenges…" formula — banned.
  • Rule of three: do NOT artificially group points into exactly three items when the
    evidence supports two or four. Lists of three feel formulaic.
  • Uniform paragraph length: vary paragraph length deliberately. One-sentence
    paragraphs are fine for emphasis. Long analytical paragraphs are fine for depth.
    Identical paragraph lengths signal AI generation.
  • Hollow hedging: "it could be argued", "one might suggest", "it seems that" —
    banned when a direct claim is supported by evidence from the paper.

Style imperatives:
  • Be direct. State the finding, then the evidence. No preamble.
  • Vary sentence length. Short sentences for conclusions. Longer sentences for
    technical reasoning. Never three sentences of identical length in a row.
  • Prefer specific, concrete language. Cite equation numbers, section headings, page
    evidence, or quoted claims — never wave at "the methodology" in the abstract.
  • Do not restate the paper's title or research question before critiquing it.
"""
