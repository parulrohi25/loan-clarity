import random
import pandas as pd
from app import simulate_scenario
from app import simulate_scenario, calculate_emi      # calculate_emi add karo

random.seed(42)    # har baar same data aaye, debugging easy rahe

def generate_dataset(n_samples=5000):
    rows = []
    for _ in range(n_samples):
        principal = random.randint(100000, 2000000)
        rate = round(random.uniform(7, 15), 2)
        tenure = random.randint(12, 120)
        base_emi, base_interest, _ = calculate_emi(principal, rate, tenure)

        # kitni EMI miss hui aur kaun se mahine
        num_missed = random.randint(0, 4)
        missed = random.sample(range(1, tenure + 1), num_missed)

        # 50% scenarios mein rate change hogi
        if random.random() < 0.5:
            rate_change_month = random.randint(2, tenure)
            new_rate = round(random.uniform(7, 15), 2)
        else:
            rate_change_month = None
            new_rate = None

        total_interest, total_months = simulate_scenario(
            principal, rate, tenure, missed, rate_change_month, new_rate
        )

        rows.append({
            # features (inputs)
            "principal": principal,
            "rate": rate,
            "tenure": tenure,
            "num_missed": num_missed,
            "rate_change_month": rate_change_month if rate_change_month else 0,
            "new_rate": new_rate if new_rate else rate,
            # targets (jo model predict karega)
            "total_interest": round(total_interest, 2),
            "total_months": total_months,
            "base_emi": round(base_emi, 2),
            "base_interest": round(base_interest, 2),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset(5000)
    df.to_csv("loan_data.csv", index=False)
    print(df.head())
    print(df.describe())