from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from predikit import ModelTool

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Wrap model with Predikit
tool = ModelTool(model=model)

# Example prediction
sample = {
    "sepal length (cm)": 5.9,
    "sepal width (cm)": 3.0,
    "petal length (cm)": 5.1,
    "petal width (cm)": 1.8,
}

prediction = tool.invoke(sample)

print(prediction)
