import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from tqdm import tqdm

start_time = time.time()

# Загрузка данных
print("Загружаем данные...")
train1 = pd.read_csv('train1.csv')
train2 = pd.read_csv('train2.csv')
test = pd.read_csv('test.csv')
submit = pd.read_csv('submit.csv')

num_cols = [f'feature_{i}' for i in range(1367)]
X2 = train2[num_cols]
y2 = train2['target'].astype(int)

# Настройки кросс-валидации
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
test_probs = np.zeros(len(test))
val_scores = []


# Аугментация
def augment_data(X_train, y_train, copies=19, noise=0.025):
    pos_idx = y_train[y_train == 1].index
    X_pos = X_train.loc[pos_idx]

    augmented = []
    for _ in range(copies):
        noise_matrix = np.random.normal(0, noise, size=X_pos.shape)
        augmented.append(X_pos + noise_matrix)

    return pd.concat([X_train] + augmented), pd.concat([y_train] + [y_train.loc[pos_idx]] * copies)


imputer = SimpleImputer(strategy='median')

# Кросс-валидация
for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(X2, y2), total=n_splits, desc="Кросс-валидация")):
    print(f"\n=== Fold {fold + 1}/{n_splits} ===")

    # Разделение train2
    X_train2, X_val2 = X2.iloc[train_idx], X2.iloc[val_idx]
    y_train2, y_val2 = y2.iloc[train_idx], y2.iloc[val_idx]

    # Объединение с train1
    train_combined = pd.concat([
        train1,
        pd.concat([X_train2, y_train2], axis=1),
    ], ignore_index=True)

    # Подготовка данных
    X_train_raw = train_combined[num_cols]
    y_train = train_combined['target'].astype(int)

    # Аугментация
    print("Аугментация данных...")
    X_train_aug, y_train_aug = augment_data(X_train_raw, y_train)

    # Импутация
    print("Импутация пропусков...")
    X_train = pd.DataFrame(imputer.fit_transform(X_train_aug), columns=num_cols)
    X_val = pd.DataFrame(imputer.transform(X_val2), columns=num_cols)
    X_test = pd.DataFrame(imputer.transform(test[num_cols]), columns=num_cols)

    # Обучение
    print("Обучение CatBoost...")
    model = CatBoostClassifier(
    iterations=10000,
    learning_rate=0.01,
    depth=15,
    l2_leaf_reg=10,
    grow_policy='Lossguide',
    eval_metric='AUC',
    task_type='GPU',
    random_seed=42,
    verbose=200,
    early_stopping_rounds=1000,
    )

    model.fit(
        X_train, y_train_aug,
        eval_set=(X_val, y_val2),
        use_best_model=True
    )

    # Валидация
    val_probs = model.predict_proba(X_val)[:, 1]
    val_score = roc_auc_score(y_val2, val_probs)
    val_scores.append(val_score)
    print(f"Fold {fold + 1} ROC AUC: {val_score:.4f}")

    # Предсказание на тесте
    test_probs += model.predict_proba(X_test)[:, 1] / n_splits

# Средний результат валидации
print(f"\nСредний ROC AUC на валидации: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")

# Формирование submission
submission = pd.DataFrame({
    'index': test['index'],
    'score': test_probs
})

final_submission = submit[['index']].merge(submission, on='index', how='left')
final_submission.to_csv('submission_final.csv', index=False)

print("Сохранено: 'submission_final.csv'")
print(f"Время выполнения: {(time.time() - start_time) / 60:.1f} мин")