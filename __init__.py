"""Decision grading: how good was each decision, and how much did it cost?

    types.py        DecisionContext (recorded during play) and GradedDecision (the result)
    reference.py    the fixed reference opponents the grades are measured against
    evaluate.py     exact expected value of every candidate action, by enumeration
    grade.py        grade a whole match; picks the candidate bet sizes worth checking
    stats.py        per-player totals and the two leaderboards (pure functions)
    store.py        MemoryGradeStore (tests) -- SqlGradeStore lives in sql_store.py
"""
