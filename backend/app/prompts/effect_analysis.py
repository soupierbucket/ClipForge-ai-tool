EFFECT_TAGGING_PROMPT = """Tag only clearly supported moments in the supplied English transcript.
Use only the supplied word indexes and timestamps. Do not infer visual events.
Allowed moment_type values: punchline, surprise, reveal, fail, emphasis, awkward_pause, dramatic.
Return up to 6 moments; return fewer when evidence is weak. Prefer a spoken word at the turn of the moment.
Return strict JSON only: {"moments":[{"word_index":0,"moment_type":"surprise"}]}.
The application will validate tags against transcript words, timing, and measured audio/pause signals."""
