# Capturing the demo GIF

The README has a commented-out slot at the very top for a short screen capture of a full run.
A 15–30s GIF/MP4 of one real run is the single highest-impact thing on the page — it shows the
disambiguation prompt, the live retrieval/enrichment logs, the ranked list, and the review
scrolling by, all without anyone installing anything.

## What to record

One clean run that hits the interesting moments. A good query is the flagship one:

```bash
python cli.py "What causes hallucinations in large language models?"
```

When the disambiguation prompt appears, type a multi-select (e.g. `1 2`) so the GIF shows that
feature. Let it run through retrieval → ranked list → the synthesis count prompt → the first
screenful of the review, then stop.

Keep it short: people watch ~10–20s. If the full run is long, either speed it up 2× in the
editor or trim to: prompt → a few seconds of logs → ranked list → top of the review.

## How to capture (Windows)

- **GIF (simplest):** [ScreenToGif](https://www.screentogif.com/) — free, records a region
  straight to `.gif`, has a built-in trimmer. Record just the terminal window.
- **MP4 → GIF:** Xbox Game Bar (`Win+G`) or OBS to record `.mp4`, then convert:
  ```bash
  ffmpeg -i demo.mp4 -vf "fps=12,scale=900:-1:flags=lanczos" -loop 0 demo.gif
  ```
  `fps=12` and width `900` keep the file small (aim for < 5 MB so it loads fast on GitHub).

## Where to put it

Save the final file as `examples/screenshots/demo.gif`, then in the top of `README.md`
uncomment the image line and delete the fallback link:

```markdown
![lr_tool demo](examples/screenshots/demo.gif)
```

## Before committing

- Make sure no API key, `.env` path, or private directory name is visible in the terminal
  (same secret-check we did for the static screenshots).
- A GIF can get large — confirm the file size is sane before `git add`.
