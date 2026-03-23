# AGENTS.md

## Plot Comparison Registry (Persistent)

Use `docs/plot_comparisons.md` as the persistent cross-session source of truth for requested plot comparisons.

When the user asks for a new plot/comparison:
- Add a new entry with the next `plot-####` id.
- Set `status: requested`.
- Fill in all known fields from the request and leave unknown fields as `TBD`.

When working on an existing plot/comparison:
- Update the existing entry instead of creating duplicates.
- Move status through: `requested` -> `in_progress` -> `implemented` -> `validated` (or `blocked`).
- Record script/notebook paths and output artifact paths in `implementation_refs` and `artifact_outputs`.

Registry rules:
- Keep historical entries; do not delete old entries.
- Use UTC dates in ISO format (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SSZ`).
- Keep entries concise, structured, and machine-parsable in markdown.
