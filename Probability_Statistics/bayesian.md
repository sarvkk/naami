# Bayesian Update in Disease Diagnosis

## Overview
This document explains the Bayesian update process in the context of medical diagnostics. I'll explore how Bayes' theorem allows to update beliefs about a patient having a disease after receiving test results.

## The Problem Statement
Medical tests are rarely perfect. Even the best diagnostic tests can sometimes produce false positives (indicating disease when there is none) or false negatives (missing disease when it is present). Doctors need to interpret test results in light of:

1. The prevalence of the disease in the population (prior probability)
2. The test's sensitivity and specificity (likelihood)

By applying Bayes' theorem, we can calculate the actual probability that a patient has the disease after observing a test result.

## Key Concepts

### Prior Probability
- The probability of having the disease before any test is performed
- Often based on disease prevalence in the relevant population
- May be adjusted based on patient risk factors (age, family history, etc.)

### Test Characteristics
- **Sensitivity**: The probability of a positive test result given that the patient has the disease [P(T+|D)]
- **Specificity**: The probability of a negative test result given that the patient does not have the disease [P(T-|¬D)]

### Bayes' Theorem
The formula for updating our belief after observing evidence:

$$P(D|T+) = \frac{P(T+|D) \times P(D)}{P(T+)}$$

Where:
- P(D|T+) is the posterior probability (probability of disease given a positive test)
- P(T+|D) is the likelihood (test sensitivity)
- P(D) is the prior probability (disease prevalence)
- P(T+) is the evidence (total probability of a positive test)

The evidence term P(T+) can be expanded using the law of total probability:

$$P(T+) = P(T+|D) \times P(D) + P(T+|¬D) \times P(¬D)$$

Where P(T+|¬D) = 1 - specificity (false positive rate)

## Implementation Details

In our implementation, we:

1. **Created a simulated dataset** of 1,000 patients with:
   - Disease prevalence of 5%
   - Test sensitivity of 95%
   - Test specificity of 90%

2. **Calculated the Bayesian update** for the overall population:
   - Prior probability P(D) = 0.05
   - Likelihood P(T+|D) = 0.95
   - Evidence P(T+) = (0.05 × 0.95) + (0.95 × 0.10) = 0.14
   - Posterior P(D|T+) = (0.05 × 0.95) / 0.14 = 0.34

3. **Extended the analysis to different age groups** with varying prevalence:
   - Young (20–40): 1%
   - Middle (40–60): 5%
   - Senior (60+): 15%

4. **Visualized prior vs posterior probabilities**:
   - Bar chart comparing P(D), P(D|T+), and P(D|T–)

5. **Computed Bayesian updates by age group**:
   - For each group, calculated P(D|T+) and P(D|T–)
   - Quantified relative increase/decrease in risk

6. **Visualized group comparisons**:
   - Grouped bar charts of prior and posterior probabilities across age brackets

7. **Evaluated the value of testing**:
   - Defined a `test_value()` function to compare strategies:
     • Treat all (no test)  
     • Treat none (no test)  
     • Test & treat positives only  
     • Test & optimal decision based on a benefit–risk threshold  
   - Incorporated treatment benefit, treatment risk, and cost of testing