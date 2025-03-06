from scripts.simulation_imports import *
from openai import OpenAI
import pandas as pd
import re


def get_slate_docs_feature(slate_indices, candidate_docs):
    return [candidate_docs[idx] for idx in slate_indices]


def get_item_ids_and_titles(item_ids, news_df):
    # Create a dictionary of itemId -> title for faster lookup
    item_to_title = dict(zip(news_df["itemId"], news_df["title"]))

    # Retrieve the titles for each item_id
    item_titles = [
        (item_id, item_to_title.get(item_id, "Title not found")) for item_id in item_ids
    ]

    return item_titles


# Function to parse the LLM's response into a list of items
def parse_recommended_slate(response_text):
    # Regex pattern to match item IDs (e.g., N82348, N18469)
    # Assumes item IDs start with "N" followed by digits
    item_id_pattern = r"N\d+"

    # Find all matches of the pattern in the response text
    item_ids = re.findall(item_id_pattern, response_text)
    # Remove single quotes from item IDs (if any)
    item_ids = [item_id.replace("'", "") for item_id in item_ids]

    # Remove duplicates while preserving order
    item_ids = list(dict.fromkeys(item_ids))

    return item_ids


# Function to process a single row and get the recommended slate
def process_row(row):
    # Get the user history for the current row's initial_user_state
    user_state_tuple = tuple(row["initial_user_state"])
    user_history = user_history_dict.get(user_state_tuple, [])

    # Get titles for candidate and slate items
    candidate_titles = get_item_ids_and_titles(candidate_ids[row.name], news_df)
    slate_titles = get_item_ids_and_titles(slate_item_ids[row.name], news_df)

    # Construct the prompt
    prompt = f"""
    You are a slate generator. Given a user's interaction history, select the best 10 items to recommend.

    User History:
    {', '.join([f"{item} ({title})" for item, title in user_history])}

    Candidate Docs:
    {', '.join([f"{item} ({title})" for item, title in candidate_titles])}

    Given Slate:
    {', '.join([f"{item} ({title})" for item, title in slate_titles])}

    What changes will you make to this slate? If you think any item needs to be changed, consider selecting it from the candidate docs. Take this decision in such a way that the user's click probability from the slate is maximized. The user is only allowed to pick one item.

    Please provide the recommended slate as a numbered list of items.
    """

    # Get the recommended slate from the LLM
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1:free", messages=messages
    )
    response_text = response.choices[0].message.content

    # Parse the response into a list of items
    recommended_slate = parse_recommended_slate(response_text)

    return recommended_slate


if __name__ == "__main__":
    DATA_PATH = Path.home() / Path(os.environ.get("RSYS_DATA", "rsys_data/rsys_2025"))
    gen_slates_dir = DATA_PATH / "gen_slates"

    # Define file path
    feather_file_path = gen_slates_dir / "wp_user_slates.feather"
    df = pd.read_feather(feather_file_path)

    # Create the new column 'slate_docs_feature'
    df["slate_docs_feature"] = df.apply(
        lambda row: get_slate_docs_feature(row["slate_docs"], row["candidate_docs"]),
        axis=1,
    )
    USER_SEED = 11

    DATA_PATH = Path.home() / Path(os.environ.get("DATA_PATH"))
    dataset_interaction_path = DATA_PATH / Path("MINDlarge_train/test_50.feather")
    interaction_data = pd.read_feather(dataset_interaction_path)
    dataset_path = DATA_PATH / Path("MINDlarge_train/test_50.feather")
    category_data = pd.read_feather(dataset_path)

    category_data["observed_state"] = category_data["observed_state"].apply(
        lambda x: tuple(x) if x is not None else ()
    )
    state_history_dict = dict(
        zip(category_data["observed_state"], category_data["click_history"])
    )

    base_path = Path.home() / Path(os.environ.get("DATA_PATH"))
    news_df = pd.read_feather(
        base_path / Path("MINDlarge_train/news_glove_embed_50.feather")
    )
    embedding_dict = dict(zip(news_df["itemId"], news_df["embedding"]))
    none_keys = [key for key, value in embedding_dict.items() if value is None]

    # Convert embeddings to tuples only for valid (non-None) values
    embedding_lookup = {
        tuple(item_embedding): item_id
        for item_id, item_embedding in embedding_dict.items()
        if item_embedding is not None
    }
    # Retrieve item IDs for candidate_docs
    candidate_ids = [
        [
            embedding_lookup.get(tuple(embedding), "Not Found")
            for embedding in candidate_list
        ]
        for candidate_list in df["candidate_docs"]
    ]

    # Retrieve item IDs for slate_docs_feature
    slate_item_ids = [
        [
            embedding_lookup.get(tuple(embedding), "Not Found")
            for embedding in slate_list
        ]
        for slate_list in df["slate_docs_feature"]
    ]

    user_history_dict = {}
    item_to_title = dict(zip(news_df["itemId"], news_df["title"]))

    for user_state in df["initial_user_state"]:
        user_state_tuple = tuple(user_state)  # Convert NumPy array to tuple

        if user_state_tuple in user_history_dict:
            continue  # Skip if already processed

        # Retrieve click history (list of item IDs) for the given user_state
        click_history = state_history_dict.get(user_state_tuple, [])

        # Retrieve corresponding titles
        history_with_titles = [
            (item_id, item_to_title.get(item_id, "Title not found"))
            for item_id in click_history
        ]

        # Store in dictionary
        user_history_dict[user_state_tuple] = history_with_titles

    candidate_titles = get_item_ids_and_titles(candidate_ids[0], news_df)
    item_impressions = get_item_ids_and_titles(slate_item_ids[0], news_df)

    # Initialize the OpenAI client
    api_key = os.getenv("DEEPSEEK_API_KEY_2")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    df["llm_slate"] = df.apply(process_row, axis=1)

    # Step 1: Group category_data by observed_state and collect all click values in a list
    state_to_clicks = defaultdict(list)
    for observed_state, click in zip(
        category_data["observed_state"], category_data["click"]
    ):
        state_to_clicks[observed_state].append(click)

    # Convert the defaultdict to a DataFrame for easier merging
    clicks_df = pd.DataFrame(
        [
            (state, click_idx, click)
            for state, clicks in state_to_clicks.items()
            for click_idx, click in enumerate(clicks)
        ],
        columns=["observed_state", "click_idx", "click"],
    )

    # Step 2: Convert df['initial_user_state'] to tuples for comparison
    df["initial_user_state_tuple"] = df["initial_user_state"].apply(tuple)

    # Step 3: Assign a sequential index to each occurrence of initial_user_state in df
    df["click_idx"] = df.groupby("initial_user_state_tuple").cumcount()

    # Step 4: Merge df with clicks_df to align click values
    df = df.merge(
        clicks_df,
        how="left",
        left_on=["initial_user_state_tuple", "click_idx"],
        right_on=["observed_state", "click_idx"],
    )

    # Step 5: Clean up the DataFrame
    df.rename(columns={"click": "original_click"}, inplace=True)
    df.drop(
        columns=["initial_user_state_tuple", "click_idx", "observed_state"],
        inplace=True,
    )

    df["hit"] = df.apply(
        lambda row: 1 if row["original_click"] in row["llm_slate"] else 0, axis=1
    )

    df["initial_user_state_tuple"] = df["initial_user_state"].apply(tuple)

    # Step 2: Group by initial_user_state and calculate the mean of 'hit' for each group
    grouped_means = df.groupby("initial_user_state_tuple")["hit"].mean().reset_index()
    grouped_means.rename(columns={"hit": "group_mean_hit"}, inplace=True)

    # Step 3: Calculate the overall average of the group means
    overall_mean = grouped_means["group_mean_hit"].mean()

    # Display the results
    print("Group-level averages:")
    print(grouped_means)
    print("\nOverall average:")
    print(overall_mean)
