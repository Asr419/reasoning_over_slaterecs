import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import random
import re
from tqdm import tqdm

load_dotenv()


def extract_news_ids(text_set):
    news_ids = set()
    pattern = re.compile(r"N\d+")  # Matches items like "N12345"

    for text in text_set:
        matches = pattern.findall(text)  # Find all occurrences of "Nxxxxx"
        news_ids.update(matches)  # Add valid IDs to the set

    return news_ids


if __name__ == "__main__":
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    base_path = Path("/home/aayush/rsys_data")
    save_dir = base_path / "rsys_2025"
    feather_path = save_dir / "interactions.feather"
    df = pd.read_feather(feather_path)

    client = OpenAI(api_key=api_key, base_url=base_url)

    news = pd.read_csv(
        base_path / Path("news.tsv"),
        sep="\t",
        names=[
            "itemId",
            "category",
            "subcategory",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities",
        ],
    )
    candidate_items = 50
    num_of_users = 1200

    # Sample 2000 unique user IDs
    random_users = random.sample(df["userId"].unique().tolist(), num_of_users)

    # Create an empty list to store results
    results = []

    # Process each user
    for user_id in tqdm(random_users):
        # Extract user's row
        user_row = df[df["userId"] == user_id].iloc[0]

        # Extract actual presented slate
        actual_presented_slate = set(user_row["presented_slate"])

        # Extract impressions (removing "-0" or "-1" suffixes)
        user_impressions = [
            imp.split("-")[0] for imp in user_row["impressions"].split()
        ]

        # Sample additional news items from news_df to make total 50
        all_news_items = set(news["itemId"]) - set(user_impressions)  # Avoid duplicates
        additional_news = random.sample(
            list(all_news_items), max(0, candidate_items - len(user_impressions))
        )  # Fill up to 50

        # Final expanded impressions
        final_impressions = list(set(user_impressions) | set(additional_news))

        # Extract titles for impressions
        impression_titles = {
            row["itemId"]: row["title"]
            for _, row in news[news["itemId"].isin(final_impressions)].iterrows()
        }

        # Extract click history (removing "-0"/"-1")
        click_history_items = user_row["click_history"]

        # Extract titles for click history
        click_history_titles = {
            row["itemId"]: row["title"]
            for _, row in news[news["itemId"].isin(click_history_items)].iterrows()
        }

        # Format click history with titles
        formatted_click_history = "\n".join(
            [
                f"{item}: {click_history_titles.get(item, 'Unknown Title')}"
                for item in click_history_items
            ]
        )

        # Format impressions with titles
        formatted_impressions = "\n".join(
            [
                f"{item}: {impression_titles.get(item, 'Unknown Title')}"
                for item in final_impressions
            ]
        )

        # Create a formatted prompt
        prompt_content = (
            f"User ID: {user_id}\n"
            f"Click History:\n{formatted_click_history}\n\n"
            f"Candidate items:\n{formatted_impressions}\n\n"
            f"Recommend the best 10 news items for this user from the set of candidate items with the Itemids from the impressions."
        )

        # Generate recommended slate from OpenAI
        messages = [{"role": "user", "content": prompt_content}]
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1-distill-qwen-32b", messages=messages
        )
        openai_response = response.choices[0].message.content
        # Extract recommended slate from OpenAI response
        recommended_slate_raw = response.choices[0].message.content.split("\n")
        recommended_slate_set = extract_news_ids(
            recommended_slate_raw
        )  # Clean and extract IDs

        # Compute matching articles
        num_matches = len(recommended_slate_set & actual_presented_slate)

        # Compute precision and recall
        precision = (
            num_matches / len(recommended_slate_set) if recommended_slate_set else 0
        )
        recall = (
            num_matches / len(actual_presented_slate) if actual_presented_slate else 0
        )

        # Append to results list
        results.append(
            {
                "userId": user_id,
                "recommended_slate": list(recommended_slate_set),
                "presented_slate": list(actual_presented_slate),
                "number_of_matches": num_matches,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "openai_response": openai_response,
            }
        )

    results_df = pd.DataFrame(results)

    print(results_df.head())

    results_df.to_csv("slate_recommendation.csv", index=False)
