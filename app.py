from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

def format_money(number):
    return f"{number:,.2f}"

def calculate_emi(principal, annual_rate, tenure_months):
    monthly_rate = annual_rate / (12 * 100)
    emi = principal * monthly_rate * (1 + monthly_rate)**tenure_months / ((1 + monthly_rate)**tenure_months - 1)
    total_payment = emi * tenure_months
    total_interest = total_payment - principal
    return emi, total_interest, total_payment

def get_amortization_schedule(principal, annual_rate, tenure_months, emi):
    monthly_rate = annual_rate / (12 * 100)
    schedule = []

    balance = principal

    for month in range(1, tenure_months + 1):
        interest_this_month = balance * monthly_rate
        principal_this_month = emi - interest_this_month
        balance = balance - principal_this_month
        if abs(balance) < 0.01:
            balance = 0

        schedule.append({
            "month": month,
            "interest_paid": format_money(interest_this_month),
            "principal_paid": format_money(principal_this_month),
            "balance": format_money(balance)
        })

    return schedule

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    schedule = None
    if request.method == 'POST':
        principal = float(request.form['principal'])
        rate = float(request.form['rate'])
        tenure = int(request.form['tenure'])
        emi, total_interest, total_payment = calculate_emi(principal, rate, tenure)
        result = {
            "emi": format_money(emi),
            "total_interest": format_money(total_interest),
            "total_payment": format_money(total_payment)
        }
        schedule = get_amortization_schedule(principal, rate, tenure, emi)
    return render_template('index.html', result=result, schedule=schedule)

interest_model = joblib.load("interest_model.pkl")
months_model = joblib.load("months_model.pkl")

FEATURES = ["principal", "rate", "tenure", "num_missed", "rate_change_month",
            "new_rate", "base_emi", "base_interest"]

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        principal = float(request.form['principal'])
        rate = float(request.form['rate'])
        tenure = int(request.form['tenure'])
        num_missed = int(request.form['num_missed'])
        rate_change_month = int(request.form['rate_change_month'] or 0)
        new_rate = float(request.form['new_rate'] or rate)

        base_emi, base_interest, _ = calculate_emi(principal, rate, tenure)

        row = pd.DataFrame([{
            "principal": principal, "rate": rate, "tenure": tenure,
            "num_missed": num_missed, "rate_change_month": rate_change_month,
            "new_rate": new_rate, "base_emi": base_emi, "base_interest": base_interest
        }], columns=FEATURES)

        if num_missed == 0 and rate_change_month == 0:
            extra_interest = 0                      # koi scenario nahi, toh extra kuch nahi
        else:
            extra_interest = interest_model.predict(row)[0]
        total_interest = extra_interest + base_interest

        if num_missed == 0:
            total_months = tenure                   # missed EMI ke bina tenure nahi badhta
        else:
            total_months = round(months_model.predict(row)[0])

        prediction = {
            "total_interest": format_money(total_interest),
            "extra_interest": format_money(extra_interest),
            "total_months": total_months,
            "extra_months": total_months - tenure
        }
    return render_template('predict.html', prediction=prediction)
    
@app.route('/compare', methods=['GET', 'POST'])
def compare():
    results = None
    if request.method == 'POST':
        # Option 1 ki values nikaalo
        principal_1 = float(request.form['principal_1'])
        rate_1 = float(request.form['rate_1'])
        tenure_1 = int(request.form['tenure_1'])
        
        # Option 2 ki values nikaalo — tum likho (same pattern, _2 wale names)
        principal_2 = float(request.form['principal_2'])
        rate_2 = float(request.form['rate_2'])
        tenure_2 = int(request.form['tenure_2'])

        # calculate_emi() ko dono ke liye call karo
        emi_1, interest_1, payment_1 = calculate_emi(principal_1, rate_1, tenure_1)
        # Option 2 ke liye bhi same — tum likho
        emi_2, interest_2, payment_2 = calculate_emi(principal_2, rate_2, tenure_2)
        
        results = {
            "option1": {"emi": format_money(emi_1), "total_interest": format_money(interest_1), "total_payment": format_money(payment_1)},
            "option2": {"emi": format_money(emi_2), "total_interest": format_money(interest_2), "total_payment": format_money(payment_2)}   # tum bharo
        }
    return render_template('compare.html', results=results)
def simulate_scenario(principal, annual_rate, tenure_months,
                      missed_payments=None, rate_change_month=None, new_rate=None):
    if missed_payments is None:
        missed_payments = []

    monthly_rate = annual_rate / (12 * 100)
    emi = calculate_emi(principal, annual_rate, tenure_months)[0]

    balance = principal
    total_interest = 0
    month = 0

    while balance > 0.01 and month < 600:      # 600 = safety cap
        month += 1

        if month == rate_change_month:
            monthly_rate = new_rate / (12 * 100)
            remaining_months = tenure_months - month + 1
            emi = calculate_emi(balance, new_rate, remaining_months)[0]

        interest_this_month = balance * monthly_rate
        total_interest += interest_this_month

        if month in missed_payments:
            principal_this_month = 0
        else:
            principal_this_month = min(emi - interest_this_month, balance)

        balance -= principal_this_month

    return total_interest, month
# print(simulate_scenario(500000, 9, 36))                                  # normal
# print(simulate_scenario(500000, 9, 36, missed_payments=[3, 7]))          # 2 EMI miss
# print(simulate_scenario(500000, 9, 36, rate_change_month=12, new_rate=11))

if __name__ == '__main__':
    app.run(debug=True)

