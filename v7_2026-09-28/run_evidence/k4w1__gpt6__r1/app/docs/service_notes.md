# solver-7b serving notes

* batch size 400 questions, pooled sampling budget 2400 (6/question), temperature fixed at 0.8
* topics are tagged upstream and are reliable
* current policy: majority vote over 6 samples of template 0 (`policy.py` in this directory)
* known issue tickets: "verifier score seems to disagree with graders on some topics" (open),
  "paraphrasing the prompt changes which wrong answer the model settles on" (open, not reproduced)
