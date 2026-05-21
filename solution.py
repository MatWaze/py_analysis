import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from pandas import DataFrame

def get_asset_level(net_worth: float):
    if pd.isna(net_worth):
        return "Неизвестно"
    elif net_worth < 100000:
        return "Низкий капитал"
    elif net_worth <= 1000000:
        return "Средний капитал"
    else:
        return "Высокий капитал"

def next_month_prediction(transactions: DataFrame):
    monthly_revenue = (transactions
        .groupby(transactions["transaction_date"].dt.to_period("M"))["amount"]
        .sum()
        .reset_index()
    )

    monthly_revenue["transaction_date"] = monthly_revenue["transaction_date"].dt.to_timestamp()

    # месяц
    x = np.arange(len(monthly_revenue)).reshape(-1, 1)

    # выручка
    y = monthly_revenue["amount"].values

    model = LinearRegression()
    model.fit(x, y)

    # Прогноз следующего месяца
    next_month_index = np.array([[len(monthly_revenue)]])
    predicted_revenue = model.predict(next_month_index)[0]

    next_month_date = monthly_revenue["transaction_date"].max() + pd.DateOffset(months=1)

    print(f"Прогноз выручки на след. месяц ({next_month_date.strftime("%m-%Y")}):{predicted_revenue:,.2f}")

# Топ-5 наиболее популярных услуг по количеству заказов
def top_popular_services(transactions: DataFrame):
    top_5_services = transactions["service"].value_counts().head(5)
    print("Топ-5 популярных услуг:")
    print(top_5_services)

# Средняя сумма транзакций по каждому городу
def city_average(transactions: DataFrame):
    avg_amount_by_city = transactions.groupby("city")["amount"].mean().sort_values(ascending=False)
    print("Средняя сумма транзакций по городам:")
    print(avg_amount_by_city)

# Услуга с наибольшей выручкой
def best_service(transactions: DataFrame):
    service_revenue = transactions.groupby("service")["amount"].sum()
    best_service = service_revenue.idxmax()
    print(f"Услуга с наибольшей выручкой: {best_service} (выручка: {service_revenue[best_service]:,.2f})")

# Процент выручки по способу оплаты
def payment_method_percentage(transactions: DataFrame):
    payment_method_pct = transactions["payment_method"].value_counts(normalize=True) * 100
    print("Процент транзакций по способам оплаты:")
    print(payment_method_pct)

# Выручка за последний месяц
def last_month_revenue(transactions: DataFrame):
    max_date = transactions["transaction_date"].max()
    last_month_start = pd.to_datetime(f"{max_date.year}-{max_date.month:02d}-01")
    last_month_revenue = transactions[transactions["transaction_date"] >= last_month_start]["amount"].sum()
    print(f"Выручка за последний месяц ({last_month_start.strftime("%m-%Y")}): {last_month_revenue:,.2f}")

# Вырочка по категориям
def asset_revenue(transactions: DataFrame):
    revenue_by_asset_level = (transactions
        .groupby("asset_level")["amount"]
        .sum()
        .sort_values(ascending=False)
    )
    print("Выручка по категориям клиентов:")
    print(revenue_by_asset_level)

def visualization(transactions: DataFrame, merged_data: DataFrame):
    # стиль
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)

    # Распределение сумм транзакций
    sns.histplot(transactions["amount"], bins=50, kde=True, ax=axes[0], color="skyblue")
    axes[0].set_title("Распределение сумм транзакций")
    axes[0].set_xlabel("Сумма")
    axes[0].set_ylabel("Частота")
    axes[0].ticklabel_format(style="plain", axis="x")

    # Диаграмма выручки по услугам
    service_revenue = transactions.groupby("service")["amount"].sum().sort_values(ascending=False)
    sns.barplot(x=service_revenue.values, y=service_revenue.index, ax=axes[1], hue=service_revenue.index, palette="viridis", legend=False)
    axes[1].set_title("Выручка по услугам")
    axes[1].set_xlabel("Сумма")
    axes[1].set_ylabel("")

    # График зависимости средней суммы транзакции от возраста клиентов
    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "от 65"]

    merged_data["age_group"] = pd.cut(
        merged_data["age"],
        bins=bins,
        labels=labels
    )

    avg_amount_by_group = (
        merged_data
        .groupby("age_group")["amount"]
        .mean()
        .reset_index()
    )

    sns.barplot(
        data=avg_amount_by_group,
        x="age_group",
        y="amount",
        ax=axes[2]
    )

    axes[2].ticklabel_format(style="plain", axis="y")
    axes[2].set_title("Средняя сумма по возрастным группам")
    axes[2].set_xlabel("Возрастная группа")
    axes[2].set_ylabel("Средняя сумма")

    plt.tight_layout()
    plt.show()

def process_data(
    transactions_path: str,
    clients_path: str
):
    try:
        transactions = pd.read_excel(transactions_path)
        clients = pd.read_json(clients_path)
    except FileNotFoundError as e:
        print(f"Ошибка загрузки файлов: {e}")
        return

    # Удаление дубликатов
    transactions.drop_duplicates(inplace=True)
    clients.drop_duplicates(inplace=True)

    # Переименование столбца для работы join
    clients.rename(columns={"id" : "client_id"}, inplace=True)

    # Убираем NaN
    transactions.dropna(inplace=True)
    clients.dropna(inplace=True)

    transactions = transactions[(transactions["amount"] > 0)]

    # Приведение дат к единому стандарту
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"], errors="coerce")

    transactions.dropna(subset=["transaction_date"], inplace=True)

    top_popular_services(transactions)
    print()

    city_average(transactions)
    print()

    best_service(transactions)
    print()

    payment_method_percentage(transactions)
    print()

    last_month_revenue(transactions)
    print()

    merged_data = pd.merge(transactions, clients, on="client_id", how="inner")
    merged_data["asset_level"] = merged_data["net_worth"].apply(get_asset_level)

    asset_revenue(merged_data)
    print()

    next_month_prediction(transactions)

    visualization(transactions, merged_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ финансовых транзакций и клиентов.")
    parser.add_argument("--transactions", type=str, required=True, help="Путь к файлу транзакций")
    parser.add_argument("--clients", type=str, required=True, help="Путь к файлу клиентов")

    args = parser.parse_args()

    process_data(transactions_path=args.transactions, clients_path=args.clients)