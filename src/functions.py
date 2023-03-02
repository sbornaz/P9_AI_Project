# Description: Functions used in the notebook

from src.global_vars import *
from src.libraries import *

## CONTENT-BASED RECOMMENDATION

# Select article id from user_id
def select_article(option, clicks, user_id):
    # Reduce dataset to user_id user
    clicks_red = clicks[clicks['user_id']==user_id]
    if len(clicks_red['click_article_id'].to_list())==0:
           return print("Zero article read. Random article recommendation")
    else:       
        if option == 'last':
            print('Last read article')

            # Get index of last click
            idx = clicks_red['click_time'].idxmax()
            print(f"Click index : {idx}")
            print(f"Click date : {clicks_red['click_time'][idx].strftime('%Y-%m-%d %H:%M:%S')}")
            # print(clicks_red) # Check sanity

            # Get article
            article_id = clicks_red['click_article_id'][idx]

            print(f"Most recent article id : {article_id}")

        elif option == 'most clicked':
            print('Most clicked article')

            # Get index of most clicked article
            articles_red = clicks_red.groupby('click_article_id').agg({
                'user_id':'count','click_time':'last',
                })
            articles_red = articles_red.rename(columns={'user_id':'total clics','click_time':'last click'})
            #print(articles_red) # Check sanity
            #print(clicks_red) # Check sanity

            # Get article
            article_id = articles_red.loc[articles_red['total clics']==articles_red['total clics'].max()]['last click'].idxmax()
            print(f"Number of clicks : {articles_red['total clics'][article_id]}")
            print(f"Article id: {article_id}")
        
        elif option == 'random':
            # Get random article from user_id clicked articles
            article_id = np.random.choice(clicks_red['click_article_id'], size=1, replace=False)[0]
            print(f"Article id: {article_id}")

             
    return article_id 

# Get 5 most similar articles using cosine similarity
def get_cosinsimilarity(df_articles_embeddings,article_id):
    # Get embedding vector for article_id
    embedding = df_articles_embeddings.loc[article_id] 
    
    # Remove embedding vector from embedding matrix
    df_articles_embeddings = df_articles_embeddings.drop(article_id)

    # Compute cosin similarity
    cosine_similarities = cosine_similarity([embedding], df_articles_embeddings)
    
    # Get indices of 5 most similar articles (sorted)
    indices = np.argsort(cosine_similarities[0])[::-1][:5]
    
    # Get corresponding article IDs
    recos = list(df_articles_embeddings.iloc[indices].index)
    return recos

# COLLABORATIVE FILTERING

def get_best_rated_reco(user_id, articles_metadata, rates, SVD_model):
    # Get all articles of the articles_metadata base (read and unread)
    recos = articles_metadata.drop(columns='created_at_ts').copy()

    # Rate for a user_id all the articles from the articles_metadata base
    recos['predicted_rate'] = recos['article_id'].apply(lambda x: SVD_model.predict(user_id, x).est)
    
    # Sort the articles by predicted rate
    recos = recos.sort_values(by='predicted_rate', ascending=False)

    # Remove the articles already read by the user
    recos = recos[~recos['article_id'].isin(rates[rates['user_id']==user_id]['click_article_id'].values)]

    # Get the top 5 articles
    top_recos = recos['article_id'][:5].values
    
    return top_recos, recos

# DATA PREPARATION
def merge_csv_files(folder_path, merged_file_name):
    """
    Args:
        folder_path (str): Le chemin d'accès au dossier contenant les fichiers CSV.
        merged_file_name (str): Le nom du fichier fusionné.

    Returns:
        Merged dataframe
    """
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    merged_clicks = pd.DataFrame()
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file.replace("\\\\", "\\"))
        merged_clicks = pd.concat([merged_clicks, df], ignore_index=True)

    merged_clicks.to_csv(merged_file_name, index=False)

    return merged_clicks

# VISUALIZATION

# Visualize boxplots
def visualize_histogram(hist_var, hist_title, hist_x_label, hist_fig): 
    # Calcul des quartiles Q1, Q2 (médiane), Q3
    q1, q2, q3 = hist_var.quantile([0.25, 0.5, 0.75])

    # Calcul des limites des whiskers
    iqr = q3 - q1  # plage interquartile
    whisker_min = q1 - 1.5 * iqr
    whisker_max = q3 + 1.5 * iqr

    # Création du boxplot avec les limites de whisker et les quartiles
    fig = plt.figure(figsize=figsize)
    
    bp = plt.boxplot(hist_var, vert=False, widths=0.5, positions=[0], whis=[1,99])
    plt.setp(bp['whiskers'], linestyle='--', color='gray', lw=2, dashes=(5, 5))
    plt.setp(bp['fliers'], marker='.', markerfacecolor='red', markersize=8, alpha=0.5)
    
    plt.hlines(y=1, xmin=whisker_min, xmax=q1, colors='b', lw=2)
    plt.hlines(y=1, xmin=q3, xmax=whisker_max, colors='b', lw=2)
    plt.hlines(y=1, xmin=q1, xmax=q3, colors='r', lw=2)
    
    plt.title(hist_title)
    plt.xlabel(hist_x_label)
    plt.yticks([1], ["Valeurs typiques"])
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    #plt.xscale('log')

    plt.show()
    fig.savefig(hist_fig, dpi=300, bbox_inches='tight')
