import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def display_pruned_df(df, row_head=2, row_tail=2, col_head=4, col_tail=3):
    first_cols = df.iloc[:, :col_head]
    last_cols = df.iloc[:, -col_tail:]
    ellipsis_col = pd.DataFrame(['...'] * len(df), columns=['...'])
    preview_cols = pd.concat([first_cols, ellipsis_col, last_cols], axis=1)

    first_rows = preview_cols.iloc[:row_head, :]
    last_rows = preview_cols.iloc[-row_tail:, :]
    ellipsis_row = pd.DataFrame([['...'] * preview_cols.shape[1]], columns=preview_cols.columns, index=['...'])
    preview = pd.concat([first_rows, ellipsis_row, last_rows], axis=0)

    class PreviewDF(pd.DataFrame):
        def _repr_html_(self):
            return super()._repr_html_() + f'{df.shape[0]} rows x {df.shape[1]} columns<br>'

    return PreviewDF(preview)


def plot_elbow_method_with_k(ax, features, k_start=1, k_end=9):
    k_range = range(k_start, k_end + 1)
    inertia = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit_predict(features)
        inertia.append(km.inertia_)

    results = pd.DataFrame({'k': list(k_range), 'inertia': inertia}, index=list(k_range))
    sns.lineplot(results, x='k', y='inertia', marker='o', ax=ax)
    ax.set_title('Elbow Point using Inertia')


def plot_silhouette_score_with_k(ax, features, k_start=2, k_end=9):
    k_range = range(k_start, k_end + 1)
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42)
        sil_score = silhouette_score(features, km.fit_predict(features))
        sil_scores.append(sil_score)

    results = pd.DataFrame({'k': list(k_range), 'silhouette': sil_scores}, index=list(k_range))
    sns.lineplot(results, x='k', y='silhouette', marker='o', ax=ax)
    ax.set_title('Silhouette Score')


def find_optimal_dbscan_params(features,
                               min_samples_range=None,
                               eps_range=None):
    if eps_range is None:
        nn = NearestNeighbors(n_neighbors=min_samples_range[0])
        nn.fit(features)
        distances, _ = nn.kneighbors(features)
        k_distances = np.sort(distances[:, -1])

        eps_min = np.percentile(k_distances, 10)
        eps_max = np.percentile(k_distances, 95)
        eps_range = np.linspace(eps_min, eps_max, 50)

    results = []
    for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
        for eps in eps_range:
            dbscan_ = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred = dbscan_.fit_predict(features)

            unique_labels = set(y_pred)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

            if n_clusters < 1:
                continue

            score = silhouette_score(features, y_pred)
            results.append({
                'min_samples': min_samples,
                'eps': eps,
                'silhouette_score': score
            })

    df = pd.DataFrame(results)

    plt.figure(figsize=(14, 4))
    sns.lineplot(df, x='eps', y='silhouette_score', hue='min_samples', marker='o', palette='deep')

    description = ('Best Parameter Pairs within \n'
                   + fr'{min_samples_range[0]}$\leq$minPts$\leq${min_samples_range[-1]}'
                   + fr' and {eps_range[0]:.4f}$\leq$eps$\leq${eps_range[-1]:.4f}')
    plt.title(description)

    return df


def find_optimal_eps(df, min_samples, min_eps=0):
    optimal_params = df[df['min_samples'] == min_samples].drop('min_samples', axis=1)

    knee_search_range = optimal_params[optimal_params['eps'] >= min_eps]
    knee_locator = KneeLocator(x=knee_search_range['eps'], y=knee_search_range['silhouette_score'],
                               curve='concave', direction='increasing')
    knee = knee_locator.knee

    plt.figure(figsize=(14, 4))
    sns.lineplot(optimal_params, x='eps', y='silhouette_score', marker='o')
    plt.axvline(knee, color='k', linestyle='--')

    eps_values = optimal_params['eps'].values
    plt.xlim((eps_values[0], eps_values[-1]))
    plt.title(f'Silhouette Scores for minPts={min_samples}')

    return knee


def plot_on_pca(ax, features, y_real, pred=None, title=''):
    pca = pd.DataFrame(PCA(n_components=2, random_state=0).fit_transform(features), columns=['pca1', 'pca2'])

    ax.set_title(title)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')

    pca['cluster'] = pred if pred is not None else y_real
    markers = style = None
    if pred is not None:
        style = 'is_correct'
        pca['is_correct'] = y_real == pred
        markers = {True: 'o', False: 'X'}

    classes = [0, 1]
    palette = sns.color_palette(n_colors=len(classes))
    colors = {cls: palette[i] for i, cls in enumerate(classes)}

    if -1 in pca['cluster'].unique():
        classes.append(-1)
        colors[-1] = (0, 0, 0)

    class_: int
    for class_, sub_df in pca.groupby('cluster'):
        sns.scatterplot(data=sub_df, x='pca1', y='pca2', style=style, markers=markers,
                        ax=ax, color=colors[class_])

    legend_elements = [
        Line2D([], [], marker='s', color='w', label=f'Class {classes[class_]}',
               markerfacecolor=colors[class_], markeredgecolor='k', markersize=6, linestyle='None')
        for class_ in classes
    ]
    if pred is not None:
        legend_elements.extend([
            Line2D([], [], marker='o', color='w', label='Correct',
                   markerfacecolor='gray', markeredgecolor='k', markersize=6, linestyle='None'),
            Line2D([], [], marker='x', color='w', label='Incorrect',
                   markeredgecolor='k', markersize=6, markeredgewidth=1.5, linestyle='None')
        ])
    ax.legend(handles=legend_elements)
