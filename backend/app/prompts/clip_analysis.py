"""Structured prompts for transcript-grounded clip analysis."""

RESPONSE_SHAPE = '''{
  "candidates": [
    {
      "start_segment": integer,
      "end_segment": integer,
      "scores": {
        "hook": number, "curiosity": number,
        "emotional_intensity": number, "novelty": number,
        "entertainment": number, "visual_potential": number,
        "payoff": number, "context": number,
        "standalone": number, "rewatch_potential": number
      }
    }
  ]
}'''

CANDIDATE_PROMPT = f"""You select short video clip ranges from a timestamped speech transcript.

SOURCE OF TRUTH
- The transcript is the only source of story facts.
- You cannot see video frames.

GROUNDING RULES
- Never invent words, events, visuals, characters, reactions, or timestamps.
- Use only the supplied integer segment IDs.
- Select contiguous inclusive ranges: start_segment <= end_segment.

SELECTION RULES
- Aim for 20-55 seconds when possible; any valid range from 1-60 seconds is acceptable.
- Prefer a clear passage with a beginning and ending.
- If there is no dramatic story, choose ordinary useful, funny, informative, or complete speech.
- Return up to the requested number of distinct candidates; avoid near-duplicates.
- Do not return an empty list merely because a passage is ordinary.

SCORING RULES
- Scores are rough transcript-based estimates from 0 to 10.
- They are not facts and do not predict views.
- Keep visual_potential conservative because frames are unavailable.

OUTPUT RULES
- Return JSON only, with no markdown or extra text.
- Follow this exact shape:
{RESPONSE_SHAPE}
"""

RANKING_PROMPT = """Rank the already validated clip ranges.

Use only the supplied candidate data. Compare clarity, completeness, pacing,
standalone context, and entertainment signals. Do not add, edit, or invent
timestamps. Do not claim to predict views.

Return JSON only:
{"ranked_ids":["candidate_id"],"recommended_id":"candidate_id"}

Include every supplied candidate ID exactly once, strongest first. Set
recommended_id to the first and strongest candidate."""
