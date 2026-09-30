# Sign off the launch of model B

Model **B** is a candidate to replace production model **A**.  Both models answered the same eval pool of
**600 prompts** (code, math, science, writing, advice).  An LLM judge graded all 1200 responses (pass/fail); its
verdicts, the prompt category and each response's length in tokens are in `/app/data/pool.csv`.

Before launch the review board needs three numbers, defined by **full verification** - the only verdict the
board accepts as ground truth (see `/app/docs/labelling.md`):

* **Q1** - B's pass rate on the pool: the fraction of B's 600 responses that are correct.
* **Q2** - pass rate of B minus pass rate of A on the pool.
* **Q3** - `SHIP` if B's pass rate is higher than A's, otherwise `HOLD`.

You can buy labels from the lab (`lab spec` shows the services and prices; `lab crowd|review|verify
model=A|B items=...`).  **Budget: 500 credits.**  Crowd votes cost 0.05, expert review 0.10 and full verification
2.00 credits per response.

## Deliverables

1. **`/app/answers.md`** with exactly these three lines (graded):

       Q1: <fraction, e.g. 0.712>
       Q2: <signed difference, e.g. -0.034>
       Q3: <SHIP or HOLD>

   Q1 and Q2 must be within 0.02 of the truth; Q3 must be right.
2. **`/app/report.md`** - a note to the review board: how you estimated the numbers and how much to trust them.
   Not scored.
