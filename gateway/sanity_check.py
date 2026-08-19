from complexity_scorer import ComplexityScorer
scorer = ComplexityScorer()
test_cases = [
    ("What is the capital of France?", "should be LOW"),
    ("Hi, how are you?", "should be LOW"),
    ("Debug this: a React useEffect hook causing an infinite render loop", "should be HIGH"),
    ("Explain step by step why the halting problem is undecidable", "should be HIGH"),
    ("Write a haiku about autumn", "should be LOW-MED"),
    ("Design a rate limiter that handles distributed nodes without a shared clock", "should be HIGH"),
]

print("Sanity-checking the pre-trained complexity model against known easy/hard prompts:\n")
for prompt, expectation in test_cases:
    score = scorer.score(prompt)
    print(f"  {score:.3f}  ({expectation:12s})  {prompt[:60]}")