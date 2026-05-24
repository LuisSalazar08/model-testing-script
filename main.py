from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

datasets = {
    'Iris': load_iris(),
    'Wine': load_wine()
}

models = {
    'Árbol de Decisión (J48/CART)': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Backpropagation (BPN)': MLPClassifier(max_iter=2000, random_state=42)
}

test_splits = [0.5, 0.2]

for data_name, data in datasets.items():
    print(f"\n{'=' * 50}\nDATASET: {data_name}\n{'=' * 50}")
    X, y = data.data, data.target

    for test_size in test_splits:
        train_pct = int((1 - test_size) * 100)
        test_pct = int(test_size * 100)
        print(f"\n--- PARTICIÓN: {train_pct}% Training / {test_pct}% Testing ---")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            err = 1 - acc
            cm = confusion_matrix(y_test, y_pred)

            print(f"\nModelo: {model_name}")
            print(f"Porcentaje de clasificación (Aciertos): {acc * 100:.2f}%")
            print(f"Error de clasificación: {err * 100:.2f}%")
            print(f"Matriz de Confusión:\n{cm}")