import pandas as pd

df = pd.read_json("per_sentence_results.jsonl", lines=True)

with pd.ExcelWriter("per_sentence_results.csv", engine="csvwriter") as writer:
    df.to_csv(writer, index=False, sheet_name="Results")
    wb  = writer.book
    ws  = writer.sheets["Results"]

    # Wrap text + top-align everywhere
    wrap = wb.add_format({"text_wrap": True, "valign": "top"})
    ws.set_column(0, len(df.columns)-1, None, wrap)

    # Heuristic column widths based on longest string in each column
    for c, col in enumerate(df.columns):
        series = df[col].astype(str)
        max_len = max([len(col)] + series.map(len).tolist())
        width = min(100, max_len + 2)  # cap super-long columns
        ws.set_column(c, c, width)

    # Optional: give rows a taller default so wrapped lines aren’t cramped
    for r in range(1, len(df) + 1):
        ws.set_row(r, 30)  # tweak 24–36 depending on your font/zoom

