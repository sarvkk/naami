import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100

# 1. Simulating data for 1000 patients, including whether they have a disease and their test results
np.random.seed(123)  # For reproducibility

n_patients = 1000
disease_prev = 0.05  # 5% of patients actually have the disease

# Set test performance parameters
sens = 0.95        # Sensitivity: how good the test is at catching the disease
specificity = 0.90 # Specificity: how good the test is at ruling out those without disease

# Randomly assign disease status based on prevalence
disease_status = np.random.choice([True, False], size=n_patients, 
                                  p=[disease_prev, 1-disease_prev])

# Generate test results based on whether or not the person has the disease
test_results = []
for has_disease in disease_status:
    if has_disease:
        test_results.append(np.random.choice([True, False], p=[sens, 1-sens]))
    else:
        test_results.append(np.random.choice([True, False], p=[1-specificity, specificity]))

# Combine into a single DataFrame
df = pd.DataFrame({
    'disease': disease_status,
    'test_positive': test_results
})

# 2. Let's calculate how many patients fall into each category
total_patients = len(df)
disease_count = df['disease'].sum()
no_disease_count = total_patients - disease_count
test_positive_count = df['test_positive'].sum()
test_negative_count = total_patients - test_positive_count

# Count how many were true/false positives and negatives
true_positive = df[(df['disease'] == True) & (df['test_positive'] == True)].shape[0]
false_positive = df[(df['disease'] == False) & (df['test_positive'] == True)].shape[0]
true_negative = df[(df['disease'] == False) & (df['test_positive'] == False)].shape[0]
false_negative = df[(df['disease'] == True) & (df['test_positive'] == False)].shape[0]

# Put the results into a neat table
contingency = pd.DataFrame({
    'Disease': [true_positive, false_positive, true_positive + false_positive],
    'No Disease': [false_negative, true_negative, false_negative + true_negative],
    'Total': [true_positive + false_negative, false_positive + true_negative, total_patients]
}, index=['Test Positive', 'Test Negative', 'Total'])

print("\nContingency Table (Patient Counts):")
print(contingency)

# 3. Let's apply Bayes' Rule to see what a positive test result really tells us

# Start with our prior belief about the disease
prior_prob_disease = disease_prev

# Likelihood: probability of a positive test if the disease is present
likelihood = sens

# Total probability of a positive test, combining true and false positives
evidence = (prior_prob_disease * likelihood) + ((1 - prior_prob_disease) * (1 - specificity))

# Here's the actual probability the patient has the disease *after* a positive test
posterior_prob_disease = (prior_prob_disease * likelihood) / evidence

print("\n--- Bayesian Update for Disease Diagnosis ---")
print(f"Prior Probability of Disease P(D): {prior_prob_disease:.4f}")
print(f"Likelihood - Sensitivity P(T+|D): {likelihood:.4f}")
print(f"Evidence - Probability of Positive Test P(T+): {evidence:.4f}")
print(f"Posterior Probability of Disease given Positive Test P(D|T+): {posterior_prob_disease:.4f}")

# What if the test is negative? What's the chance the person still has the disease?
likelihood_negative = 1 - sens
evidence_negative = (prior_prob_disease * likelihood_negative) + ((1 - prior_prob_disease) * specificity)
posterior_prob_disease_negative = (prior_prob_disease * likelihood_negative) / evidence_negative

print(f"Posterior Probability of Disease given Negative Test P(D|T-): {posterior_prob_disease_negative:.4f}")

# 4. Let's visualize how the test results change our belief about the disease
labels = ['Prior P(D)', 'Posterior P(D|T+)', 'Posterior P(D|T-)']
values = [prior_prob_disease, posterior_prob_disease, posterior_prob_disease_negative]
colors = ['lightblue', 'green', 'red']

fig, ax = plt.subplots()
bars = ax.bar(labels, values, color=colors)
ax.set_ylabel('Probability')
ax.set_title('Bayesian Update: Effect of Test Results on Disease Probability')
ax.set_ylim(0, 1.0)

# Add labels above the bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 5. Suppose disease prevalence differs by age. How does that affect test interpretation?
age_groups = ['Young (20-40)', 'Middle (40-60)', 'Senior (60+)']
prior_by_age = [0.01, 0.05, 0.15]  # Disease is more common in older patients
posteriors_positive = []
posteriors_negative = []

print("\n--- Bayesian Updates by Age Group ---")
for i, age_group in enumerate(age_groups):
    prior = prior_by_age[i]
    
    # Calculate posterior for positive test
    evidence_pos = (prior * sens) + ((1 - prior) * (1 - specificity))
    posterior_pos = (prior * sens) / evidence_pos
    posteriors_positive.append(posterior_pos)
    
    # And for a negative test
    evidence_neg = (prior * (1 - sens)) + ((1 - prior) * specificity)
    posterior_neg = (prior * (1 - sens)) / evidence_neg
    posteriors_negative.append(posterior_neg)
    
    print(f"\nAge Group: {age_group}")
    print(f"Prior P(D): {prior:.4f}")
    print(f"Posterior P(D|T+): {posterior_pos:.4f}")
    print(f"Posterior P(D|T-): {posterior_neg:.4f}")
    print(f"Positive Test Increases Probability by: {(posterior_pos - prior)/prior*100:.1f}%")
    print(f"Negative Test Decreases Probability by: {(prior - posterior_neg)/prior*100:.1f}%")

# 6. Visualize how much the test helps across age groups
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(age_groups))
width = 0.25

ax.bar(x - width, prior_by_age, width, label='Prior P(D)', color='lightblue')
ax.bar(x, posteriors_positive, width, label='Posterior P(D|T+)', color='green')
ax.bar(x + width, posteriors_negative, width, label='Posterior P(D|T-)', color='red')

ax.set_ylabel('Probability')
ax.set_title('Bayesian Update by Age Group: Effect of Test Results on Disease Probability')
ax.set_xticks(x)
ax.set_xticklabels(age_groups)
ax.legend()
ax.set_ylim(0, max(posteriors_positive) * 1.1)

for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3)

plt.tight_layout()
plt.show()

# 7. Let's estimate the value of doing the test in the first place
def test_value(prior, sens, specificity, treatment_benefit, treatment_risk, cost_of_test):
    """
    Calculate how useful testing is, in terms of expected value.
    """
    evidence_pos = (prior * sens) + ((1 - prior) * (1 - specificity))
    posterior_pos = (prior * sens) / evidence_pos
    
    evidence_neg = (prior * (1 - sens)) + ((1 - prior) * specificity)
    posterior_neg = (prior * (1 - sens)) / evidence_neg
    
    p_test_pos = evidence_pos
    p_test_neg = 1 - evidence_pos
    
    # Strategy 1: Treat everyone without testing
    ev_treat_all = prior * treatment_benefit - (1 - prior) * treatment_risk
    
    # Strategy 2: Do nothing
    ev_treat_none = 0  # Just baseline, no gains or losses
    
    # Strategy 3: Test, and treat only if test is positive
    ev_test_treat_positive = (p_test_pos * (posterior_pos * treatment_benefit - 
                                            (1 - posterior_pos) * treatment_risk)) - cost_of_test
    
    # Strategy 4: Make optimal decision based on posterior probability
    threshold = treatment_risk / (treatment_benefit + treatment_risk)
    
    if posterior_pos > threshold:
        ev_test_optimal = ev_test_treat_positive
    else:
        ev_test_optimal = -cost_of_test 
    
    return {
        'Treat All (No Test)': ev_treat_all,
        'Treat None (No Test)': ev_treat_none,
        'Test & Treat Positive': ev_test_treat_positive,
        'Test & Optimal Decision': ev_test_optimal,
        'Posterior Given Positive': posterior_pos,
        'Posterior Given Negative': posterior_neg,
        'Decision Threshold': threshold
    }

# Testing across different age groups
treatment_benefit = 100  
treatment_risk = 20      
test_cost = 10           

print("\n--- Value of Testing by Age Group ---")
for i, age_group in enumerate(age_groups):
    prior = prior_by_age[i]
    results = test_value(prior, sens, specificity, 
                         treatment_benefit, treatment_risk, test_cost)
    
    print(f"\nAge Group: {age_group}, Prior: {prior:.4f}")
    print(f"Decision Threshold: {results['Decision Threshold']:.4f}")
    print(f"Posterior|T+: {results['Posterior Given Positive']:.4f}, Posterior|T-: {results['Posterior Given Negative']:.4f}")
    
    for strategy, value in results.items():
        if strategy not in ['Posterior Given Positive', 'Posterior Given Negative', 'Decision Threshold']:
            print(f"{strategy}: {value:.2f}")
